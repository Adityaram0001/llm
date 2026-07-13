# Wave B — positional encodings (learned / sinusoidal / RoPE / ALiBi / NoPE) + the length-extrapolation probe: the complete deep dive

*Discussion session 2026-07-13, consolidating Wave B (run 2026-07-12 on the RTX 5090, run_ids
`20260712_p5_s-wave-b-*`). This note pulls everything from the 4 runs' `notes.md`,
`docs/results/ablation_log.md`, `docs/results/wave_b_positional_encodings.png`, the RW-5 code fix,
`scripts/eval_extrapolation.py`, and D-037 into one revision-ready place — with the mechanisms of
all five encodings spelled out, the extrapolation probe explained as a *method* not just a result,
and the two genuine surprises (sinusoidal, ALiBi) given proper airtime. No code or specs were
changed in this discussion session (RW-5's `forward()` fix was part of the earlier *implementation*
session, logged in D-037); future-action items live in D-037's "Revisit if" and RW-5's open half.*

---

## TL;DR (the one-screen revision box)

Wave B asked: **how does the model learn about token order, and which mechanism is best — both at
the length it trains on AND at lengths it's never seen?** Five encodings, single-variable swaps off
the true baseline (`20260711_p4_s-baseline`, RoPE, val 3.5037), same seed 1337, 98.3M tokens, seq
512, noise floor 0.0150 (D-035).

**At the trained length (512):**

| encoding | mechanism family | final val loss | Δ vs RoPE | extra params |
|---|---|---|---|---|
| **ALiBi** | in-attention bias | **3.4830** | **−0.0207** | 0 |
| RoPE (baseline) | in-attention rotation | 3.5037 | — | 0 |
| NoPE | none (causal mask only) | 3.6997 | +0.1960 | 0 |
| learned | additive-at-input | 3.7311 | +0.2274 | +98,304 |
| sinusoidal | additive-at-input | 4.9896 | +1.4859 | 0 |

**The length-extrapolation probe (train@512, eval ppl@512/1024/2048) — the headline:**

| encoding | ppl @512 | ppl @1024 | ppl @2048 | behavior |
|---|---|---|---|---|
| **ALiBi** | 32.56 | 32.08 | **31.67** | **IMPROVES with context** |
| RoPE | 33.24 | 36.79 | 45.68 | degrades gracefully |
| NoPE | 40.43 | 67.18 | **731.91** | collapses |
| learned / sinusoidal | 41.73 / 146.87 | **ValueError** | **ValueError** | physically cannot run |

- **ALiBi beats RoPE at trained length AND improves as context grows** — the cleanest reproduction
  of a paper's headline claim in the project so far.
- **Sinusoidal is a genuine surprise-worst**, losing to even *learned* by +1.26 — losing the
  *learnable* position parameters cost far more than theory predicts at this scale.
- **NoPE trains adequately at 512 but has zero out-of-distribution length ability** (ppl 40→732).
- **learned/sinusoidal can't extrapolate *at all*** — a hard architectural ceiling (finite table),
  not a training artifact. This required the RW-5 `forward()` fix to even *express* the probe.
- **Fed to phase 9:** RoPE (current default) is solid, but **ALiBi is a real contender** for any
  long-context need (RW-5's capstone chat-context goal) — its extrapolation is strictly better here.

---

## 1. What Wave B is really testing: two questions, not one

Positional encoding is unusual among ablations because it has **two** success axes that can
disagree:

1. **Quality at the trained length** — does the model learn better *with 512-token windows* under
   this encoding?
2. **Length generalization** — trained at 512, does it still work at 1024/2048 it never saw?

RoPE and ALiBi exist almost entirely *for axis 2*. A pure axis-1 comparison would miss their whole
selling point, which is why the phase-5 spec pairs Wave B with a dedicated length-extrapolation
probe. Keep both axes separate when reading results — an encoding can win one and lose the other
(learned is fine-ish at 512 and *impossible* past it).

Like Wave A, this is a **config-only wave off the true 3-head baseline** (single variable = the
`pos_encoding` field), so every delta is a clean comparison to `p4_s_baseline` (3.5037) — no
substitute control needed (contrast Wave C's forced n_heads=4). The one exception: it needed a
small **code** enabler first (§5).

---

## 2. The five mechanisms, precisely (and their param consequences)

There are three distinct *families*, and the family determines both how position information enters
the model and whether it can extrapolate.

### 2.1 Additive-at-the-input: learned & sinusoidal
A `(seq_len, d_model)` positional vector is **added once** to the token embedding before block 0.
The model must *learn to recover* relative position from these absolute vectors deep inside.

- **learned** (GPT-1/GPT-2): a trainable lookup table, one row per absolute position up to
  `max_seq_len`. **Adds params:** `max_seq_len × d_model = 512 × 192 = 98,304` (→ 9,811,776 vs
  baseline 9,713,472 ✓). Because the table has exactly 512 rows, position 512+ **does not exist** —
  a hard ceiling.
- **sinusoidal** (Vaswani '17): a *fixed* sin/cos table, same shape, **no trainable params**
  (registered buffer → 9,713,472, identical to baseline ✓). Same 512-row ceiling.

### 2.2 In-attention, position derived per forward pass: RoPE & ALiBi
No vector added at the input. Instead each block re-derives position *inside attention* from Q/K
directly — which is exactly why these can run at **any** length (nothing is table-bounded). **Zero
extra params** for both.

- **RoPE** (Su '21): **rotates** each (q, k) pair by an angle proportional to the token's absolute
  position. The magic: the dot product of two rotated vectors depends only on their position
  *difference* — so absolute rotations produce *relative* attention. `theta=10000` sets how fast the
  rotation frequency decays across dimensions.
- **ALiBi** (Press '21): adds a **linear distance penalty** to attention logits — `score(i,j) −=
  slope_h · (i − j)`, a per-head fixed slope, no learned position at all. Recent keys are penalized
  less; far keys more. (Slopes are a geometric sequence per head; for our `n_heads=3`, which isn't a
  power of 2, the code uses the paper's interpolation fallback → slopes `[0.0625, 0.0039, 0.25]`,
  vs the clean `[0.5, 0.25, 0.125, …]` you'd get at n_heads=8.)

### 2.3 Nothing at all: NoPE
`pos_encoding="none"` — no additive vector, no in-attention position signal. The **only** thing
telling the model about order is the **causal mask** (token *t* can attend to ≤ *t*). Some research
suggests decoder-only causal attention can *implicitly* recover a notion of position from this
alone. Wave B tests whether that implicit signal generalizes (it doesn't; §4.3).

---

## 3. The trained-length results, read carefully

### 3.1 ALiBi > RoPE at 512 (real, if marginal)
ALiBi's −0.0207 is just past the noise floor and consistent from early training — so it's **real**,
though small; the disciplined statement is "ALiBi is at least as good as RoPE at trained length,
probably slightly better." The *big* ALiBi story is extrapolation (§4), not this.

### 3.2 The sinusoidal surprise (+1.486 — worst of the wave, worse than learned)
Going in, learned and sinusoidal were assumed roughly equivalent: same "additive absolute position"
family, sinusoidal just trades the 98K learnable table for a fixed one. The expectation was a small
gap. Instead sinusoidal (4.9896) lost to learned (3.7311) by **+1.26** — a chasm.

**Honest speculation for *why*** (flagged as speculation — one seed, S-tier):
- With only `d_model=192` and a *fixed* sin/cos pattern, the model has no way to *adapt* its
  position representation to what it actually finds useful — it must consume the pattern as-is and
  fold it into representations through the network. A *learned* table can shape itself into whatever
  form is easiest for the model to exploit; the fixed table cannot.
- Sinusoidal is also summed into the token embedding *before any processing*, diluting a small
  192-dim signal, whereas RoPE/ALiBi act *directly inside attention* where position is actually
  used. Learned at least gets to *learn around* that dilution; sinusoidal is stuck with it.

The takeaway isn't "sinusoidal is bad" universally — it's "at small scale, the ability to *learn*
the position representation is worth a lot," more than the classic Vaswani framing would suggest.
Flagged for re-check at M/L tier (see §6): the gap may shrink with more capacity, and our
hyperparameters were tuned around RoPE, which may quietly disadvantage sinusoidal.

### 3.3 NoPE at 512: adequate (+0.196)
NoPE lands near learned — worse than RoPE, but *functional*. At the trained length, the implicit
positional signal from the causal mask alone is enough for this task. The failure is entirely about
lengths it never saw (§4.3).

---

## 4. The length-extrapolation probe — the heart of the wave

This is where RoPE/ALiBi earn their existence and the additive-family hits a wall.

### 4.1 ALiBi *improves* with context (the clean paper reproduction)
```
ALiBi:  ppl 32.56 (512) → 32.08 (1024) → 31.67 (2048)   ↓ better with more context
```
This is textbook ALiBi and the most satisfying result of the wave. **Why it improves rather than
just holding:** ALiBi's penalty is a smooth linear function of distance with *no learned,
length-specific parameters and no frequency table* — there is literally nothing that can be
"out of distribution" at a new length. More context = strictly more (recency-weighted) evidence for
predicting the next token, and nothing to unlearn. The recency bias means far-away tokens are
gently down-weighted automatically, so extra context never *hurts*.

### 4.2 RoPE degrades gracefully
```
RoPE:   ppl 33.24 (512) → 36.79 (1024) → 45.68 (2048)   ↑ worse, but not catastrophic
```
RoPE *does* generalize (it's relative), but its rotation frequencies were only ever *seen* up to
512 positions of separation during training. At 1024/2048 the model encounters rotation angles
(large relative distances) it never trained on, so quality erodes — gracefully, because the
mechanism is continuous, but it erodes. (This is exactly what "theta scaling" / positional
interpolation techniques later fix — out of scope here, but this is the problem they solve.)

### 4.3 NoPE collapses
```
NoPE:   ppl 40.43 (512) → 67.18 (1024) → 731.91 (2048)   ✗ catastrophic
```
The sharpest degradation of any encoding. Whatever weak positional signal NoPE extracts from the
causal mask is **tightly bound to the specific 512-length window it trained on** — it is not a
generalizable notion of relative position. Push it to 2× or 4× the trained length and it falls
apart entirely (ppl 732 ≈ near-random). NoPE and ALiBi sit at *opposite ends* of the same probe
despite being one config field apart: ALiBi improves, NoPE implodes.

### 4.4 learned & sinusoidal: they can't even run
```
learned / sinusoidal:  ppl at 512 → ValueError at 1024 and beyond
```
Not "they extrapolate badly" — they **physically cannot produce an output** past position 512. Their
positional table has exactly 512 rows; asking for position 512 indexes off the end. This is a
**hard architectural ceiling**, not a training/quality artifact, and it's *categorically* different
from RoPE/NoPE degrading. Our probe surfaces it as a clean, intentional `ValueError` rather than a
silent garbage output — which required the code change in §5.

---

## 5. The enabler: RW-5's `forward()` fix (why a probe needed a code change)

The probe is "eval-only forward passes at seq_len > the trained max_seq_len." But `GPT.forward()`
originally **hard-rejected any T > max_seq_len for every encoding** — so the probe couldn't even be
*expressed*. This was RW-5, and Wave B is where its first half was resolved (D-037).

The fix is precise, not a blanket removal — it encodes exactly the §2 family distinction:

```python
if T > self.cfg.max_seq_len and self.cfg.pos_encoding in ("learned", "sinusoidal"):
    raise ValueError(...)   # only the table-bounded encodings are physically limited
# rope / alibi / none derive position on the fly → allowed at any T
```

Two supporting artifacts came with it:
- **`tests/test_model.py`** split the single old guard test into two parametrized tests —
  `..._raises_for_bounded_encodings` (learned/sinusoidal must still raise) and
  `..._allowed_for_unbounded_encodings` (rope/alibi/none must succeed past max_seq_len). A
  *deliberate* behavior change, made visible in the tests, not silent.
- **`scripts/eval_extrapolation.py`** — a new, permanent, reusable tool: load a run's checkpoint +
  config, build a fresh `MixedSourceLoader` at any seq_len against `val_sources`, report val
  loss/ppl or the expected `ValueError`. Smoke-tested against the baseline RoPE checkpoint: it
  reproduced the registered 3.5037 at 512 *exactly*, which is the sanity check that the loader/eval
  path is doing the same computation the training run did.

**The lesson:** sometimes an experiment needs a small, well-scoped code enabler before it can run.
The discipline is to make that fix *narrow* (per-encoding guard, not "remove all limits"), *tested*
(the two-way parametrized test), and *reusable* (a script, not an inline one-off) — and to log it as
part of a decision (D-037) rather than sneak it in.

---

## 6. What Wave B does *not* establish (and when to revisit)

- **S-tier only.** 10M params, 98.3M tokens, seq 512, one seed. The rankings could shift at M/L
  scale — **especially sinusoidal**, whose surprise-worst result is the most likely to be a
  small-scale and/or unfair-hyperparameter artifact (D-021's lr/schedule were tuned around RoPE).
  Re-run the trained-length comparison at M-tier before trusting the sinusoidal number.
- **The extrapolation probe used the same tuned checkpoints**, so it's a fair *relative* comparison
  but the absolute ppl-at-2048 numbers are specific to this corpus/model.
- **ALiBi ran at `n_heads=3`** (non-power-of-2 → interpolated slopes). At a power-of-2 head count the
  slopes are cleaner; unlikely to change the ranking but worth noting if ALiBi is chosen at a
  different head geometry.
- **This is the S-tier evidence for a phase-9 decision, not the decision itself.** RW-5's second
  half — the L-tier capstone's `max_seq_len` (likely ~2048 native) *and now* RoPE-vs-ALiBi choice —
  is still open, to be made when configuring the capstone. Wave B hands that decision real data
  (ALiBi extrapolates best) but doesn't pre-empt it.
- **No new rework rows here.** RW-5's remaining half and all revisit items are already tracked
  (D-037 "Revisit if", RW-5 in PROGRESS.md); per the discussion-session rule this note logs none.

---

## 7. Connections (where Wave B plugs into the rest of the project)

- **Wave A:** the other config-only wave. Together they refine the baseline recipe (Wave A: add
  qk_norm; Wave B: RoPE stays default, ALiBi noted as long-context contender). Note qk_norm and the
  positional choice are orthogonal and would stack.
- **Wave C (MLA):** MLA *is* a positional decision in disguise — its **decoupled RoPE** is the only
  positional mechanism inside an MLA block, so Wave B's "RoPE degrades gracefully" characterizes
  MLA's extrapolation too. An MLA-with-ALiBi variant is non-standard and we didn't build it; if the
  capstone wants both MLA's cache win *and* ALiBi's extrapolation, that's an open design question.
- **RW-5 / phase-9 capstone:** the user wants the final model to "carry small chats that make sense,"
  which needs real ~2k context. Wave B's probe is the direct evidence for that decision: train the
  capstone natively at ~2048 (the "real, not extrapolated" half of RW-5), and seriously consider
  ALiBi over RoPE given its strictly-better extrapolation here — or at minimum apply positional
  interpolation to RoPE.
- **Phase 6 (eval):** the extrapolation probe (`eval_extrapolation.py`) is a reusable eval harness —
  any future "does quality hold at length L" question can use it.
- **Papers:** Vaswani '17 (sinusoidal), Su '21 (RoPE), Press '21 (ALiBi), plus the NoPE line of work
  (Kazemnejad '23, "The Impact of Positional Encoding on Length Generalization").

---

## 8. Learning checkpoints (answered)

1. **What are the three positional-encoding families and how does each enter the model?**
   (a) additive-at-input (learned, sinusoidal): a `(seq_len, d_model)` vector added once before
   block 0 — model must recover relative position deep inside; table-bounded. (b) in-attention
   (RoPE rotation, ALiBi bias): position derived per forward pass inside attention from q/k — no
   table, any length. (c) none (NoPE): only the causal mask signals order.
2. **Why can RoPE/ALiBi/NoPE run past `max_seq_len` but learned/sinusoidal cannot?** The first three
   compute position on the fly (a rotation angle, a distance penalty, or nothing) which is defined
   at any position; the last two index a table with exactly `max_seq_len` rows, so position
   `max_seq_len` and beyond simply doesn't exist — a hard architectural ceiling (RW-5 guard).
3. **Why does ALiBi's ppl *improve* with more context while RoPE's degrades?** ALiBi's linear
   distance penalty has no length-specific learned parameters and nothing that can be out of
   distribution, so extra (recency-weighted) context is pure added evidence. RoPE's rotation
   frequencies were only *seen* up to 512 positions of separation in training, so large relative
   distances at 1024/2048 are unfamiliar and erode quality (gracefully, since the mechanism is
   continuous).
4. **What was the sinusoidal surprise, and the honest explanation?** Sinusoidal (+1.486) lost even
   to learned (+0.227) by a wide margin — unexpected for "same family, just fixed vs learned." Likely
   because at small `d_model` the model benefits a lot from being able to *learn/adapt* its position
   representation, which a fixed sin/cos table forbids; also flagged as possibly unfair
   (RoPE-tuned hyperparameters) and worth an M-tier re-check.

---

## 9. Key takeaways to carry forward

1. **Positional encoding has two success axes** — quality at trained length AND length
   generalization — and they can disagree. Judge both; never let a pure trained-length comparison
   stand in for the extrapolation question.
2. **ALiBi is the standout:** ≥ RoPE at 512 and its ppl *improves* to 2048 (31.67) while RoPE
   degrades (45.68) and NoPE collapses (732). Cleanest paper reproduction in the project — real
   ammunition for the capstone's long-context decision.
3. **RoPE (our default) is a safe, graceful choice** but not the extrapolation champion; if the
   capstone leans hard on long context, ALiBi (or RoPE + positional interpolation) deserves a real
   look.
4. **Additive-at-input encodings hit a hard wall** — learned/sinusoidal *cannot run* past their
   table length; this is categorically different from graceful degradation, and worth internalizing.
5. **Small scale rewards *learnable* position** — the sinusoidal surprise says the ability to adapt
   the position representation is worth more than theory predicts at 10M params (flagged for M/L
   re-check).
6. **An experiment can need a code enabler** — RW-5's narrow, tested, reusable `forward()` fix +
   `eval_extrapolation.py` is the model for that: scope it tightly, test the behavior change, make
   it reusable, log it in a decision.

**Open questions for later:** does sinusoidal's collapse persist at M/L tier or was it a small-scale
/ hyperparameter artifact? Should the L-tier capstone train natively at ~2048 with ALiBi rather than
RoPE? Both deferred to the phase-9 capstone positional/context decision (D-037, RW-5's open half).
