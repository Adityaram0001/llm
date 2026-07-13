# Wave A — norms & activations (RMSNorm/LayerNorm, pre/post-norm, SwiGLU/GELU, QK-norm): the complete deep dive

*Discussion session 2026-07-13, consolidating Wave A (run 2026-07-12 on the RTX 5090, run_ids
`20260712_p5_s-wave-a-*`). This note pulls everything from the 4 runs' `notes.md`,
`docs/results/ablation_log.md`, `docs/results/wave_a_norms_activations.png`, and D-036 into one
revision-ready place — with the parameter arithmetic worked out, the failure modes made explicit,
and the "why" behind each result spelled out beyond what the per-run notes contain. No code or
specs were changed (discussion-session rule); future-action items already live in D-036's
"Revisit if".*

---

## TL;DR (the one-screen revision box)

Wave A asked: **do the four "modernization" choices baked into our baseline (RMSNorm, pre-norm,
SwiGLU, and the option of QK-norm) actually earn their place — and by how much?** All 4 runs are
single-variable swaps off the real S-tier baseline (`20260711_p4_s-baseline`, val 3.5037), same
seed (1337), 98.3M tokens, judged against the seed-noise floor **0.0150** (D-035).

| swap | final val loss | Δ vs baseline | vs noise floor | verdict |
|---|---|---|---|---|
| **+ QK-norm** | **3.4414** | **−0.0622** | 4× floor, widening | **real win — best of wave** |
| RMSNorm → LayerNorm | 3.4878 | −0.0158 | ~1× floor | borderline — a wash |
| baseline (control) | 3.5037 | — | — | rmsnorm/pre/swiglu |
| SwiGLU → GELU (param-matched) | 3.6764 | +0.1727 | 11× floor | real — SwiGLU wins |
| pre-norm → post-norm | 6.8810 | +3.3773 | 225× floor | catastrophic (by design) |

- **QK-norm is a genuine, robust win** even at S-tier/15 layers, contrary to the going-in
  expectation that it "only matters at scale." Recommend adding it as a new default.
- **SwiGLU beats GELU** at matched matrix-params by ~0.17 val loss — a clean confirmation.
- **Pre-norm is non-negotiable at this depth** — post-norm doesn't diverge, it *stagnates*
  (a more instructive failure than "instability"; see §4).
- **RMSNorm ≈ LayerNorm** on quality — kept for its lower compute/param cost, not because
  LayerNorm was beaten.
- **This was the first real use of the noise floor**, and it immediately mattered: without it,
  LayerNorm's −0.0158 would read as "LayerNorm wins" instead of the honest "too close to call."

---

## 1. What Wave A is really testing: the modern-transformer default stack

Our baseline model (D-016) is a bundle of post-2019 choices that most people adopt *together*
without ever isolating them: RMSNorm instead of LayerNorm, pre-norm instead of post-norm, SwiGLU
instead of a GELU MLP, GPT-2-style residual init. Wave A un-bundles them and puts a number on each,
plus tests one option we left *off* by default (QK-norm). That's the whole point of an ablation
lab: not "does the modern stack work" (it obviously does — the baseline hit ppl 33) but "**which
part of it is doing the work, and how much?**"

Two framing facts that make Wave A the "easy" wave mechanically (and let it and Wave B both fit in
one session):

1. **Zero new model code.** Every axis — `norm`, `norm_position`, `ffn`, `qk_norm` — was already a
   `ModelConfig` field wired through `norms.py`/`block.py`/`ffn.py`/`attention.py` in phase 3. Wave
   A is 4 config files + 4 runs + analysis. Contrast Wave C (MLA), which needed a real
   implementation.
2. **Clean single-variable comparisons off the *true* baseline.** Unlike Wave C — which had to
   abandon the 3-head baseline because GQA-2 is undefined at `n_heads=3` (3 is prime) and build its
   own 4-head control — every Wave A run changes exactly one field and compares straight to
   `p4_s_baseline` (3.5037). No confounds, no substitute control.

---

## 2. RMSNorm vs LayerNorm: the "re-centering barely matters" result, with the exact param cost

### The mechanism
LayerNorm (Ba '16) does two things to an activation vector: **re-center** (subtract the mean) and
**re-scale** (divide by the standard deviation), then apply a learned gain *and bias*. RMSNorm
(Zhang & Sennrich '19) drops the mean-centering and the bias entirely — it just divides by the
root-mean-square and applies a learned gain:

```
LayerNorm(x) = (x − mean(x)) / std(x)  · γ + β        # 2 learned vectors: γ, β
RMSNorm(x)   =  x / sqrt(mean(x²) + ε)  · γ            # 1 learned vector: γ
```

The claim RMSNorm rides on: for transformers, the **re-scaling** is what stabilizes gradients
through depth; the **re-centering** turns out to be nearly irrelevant. If true, RMSNorm gets ~the
same effect for less compute and fewer parameters.

### What we measured
LayerNorm finished at **3.4878 vs 3.5037** — a −0.0158 edge that sits right on the 0.0150 noise
floor, and (reading the per-checkpoint trajectory, not just the endpoint) hovers between −0.014 and
−0.027 across the whole second half without growing. **This is a wash**, and it's the textbook
result: RMSNorm *matches* LayerNorm, it doesn't lose to it. (If anything LayerNorm is a hair ahead
here, but "a hair, at the noise floor, on a single seed" is exactly what you must not over-read —
the disciplined statement is "indistinguishable.")

### The exact cost RMSNorm saves (worked)
Our model has **31 norm layers**: 2 per block (attn_norm + ffn_norm) × 15 blocks + 1 final norm.
Each operates on `d_model = 192`. LayerNorm's extra **bias** vector costs one `d_model` per norm:

```
extra params (LayerNorm) = 31 norms × 192 = 5,952
→ RMSNorm model 9,713,472  vs  LayerNorm model 9,719,424   (diff = 5,952 ✓)
```

At S-tier that's 0.06% of the model — negligible in params. But the real saving is **compute**: no
mean subtraction, no bias add, per token per layer, and it scales with `d_model` at larger tiers.
**Verdict:** keep RMSNorm — same quality, cheaper. This is the one Wave A result where the noise
floor changed the conclusion from "LayerNorm wins by 0.016" to "they're equal, decide on cost."

---

## 3. SwiGLU vs GELU: the param-matching arithmetic (and the honest bias asterisk)

### The mechanism
A plain FFN is `Linear → GELU → Linear` (2 matrices). SwiGLU (Shazeer '20, "GLU Variants Improve
Transformers") replaces the single nonlinearity with a **gated** one: two parallel projections, one
passed through SiLU and multiplied elementwise into the other, then projected back (3 matrices):

```
GELU MLP:  y = W_out · GELU(W_in · x)                          # 2 matrices
SwiGLU:    y = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )      # 3 matrices
```

The gate lets the network *modulate* information flow multiplicatively, which most published
ablations show is worth a small but consistent loss improvement.

### The param match — why ffn_mult goes 8/3 for SwiGLU, 4 for GELU
SwiGLU has 3 weight matrices vs GELU's 2, so to compare **at equal parameters** you shrink SwiGLU's
hidden width. With hidden = `m · d_model`:

```
GELU   matrix params/layer = 2 · d · (m·d) = 2m · d²      # want this
SwiGLU matrix params/layer = 3 · d · (m·d) = 3m · d²
```

Set both to the conventional **8·d²**: GELU needs m = 4 (→ hidden 768), SwiGLU needs m = 8/3
(→ hidden 512). Worked for d_model=192:

```
SwiGLU: 3 · 192 · 512 = 294,912  = 8·192²  ✓
GELU:   2 · 192 · 768 = 294,912  = 8·192²  ✓        (matrix params identical)
```

**The honest asterisk:** our `GELUMLP` uses `nn.Linear` with default `bias=True`, while `SwiGLUMLP`
uses `bias=False`. So GELU carries a small bias overhead SwiGLU doesn't: (768 + 192) × 15 = 14,400
params, making the GELU *model* 9,727,872 vs SwiGLU's 9,713,472. That's **+0.15%** — negligible, and
it makes GELU *bigger*, so it can't explain GELU being *worse*. The matrices are matched exactly;
only tiny bias vectors differ. Worth stating precisely rather than claiming a perfect match.

### What we measured
GELU finished **3.6764 vs 3.5037** — **+0.1727**, ~11× the noise floor, and consistent (~+0.15 to
+0.20) from step 200 onward, not a late artifact. **SwiGLU wins cleanly and robustly**, confirming
the literature and validating D-016's default. (Speed: both ran at similar tok/s on the 5090 —
SwiGLU's third matmul is on a narrower hidden dim, so wall-clock is a wash; this is a pure quality
win, no fixed-compute tradeoff to untangle.)

---

## 4. Pre-norm vs post-norm: stagnation, not divergence (the instructive failure)

### The mechanism
Where the norm sits relative to the residual add is the whole difference:

```
pre-norm  (GPT-2+):  x = x + Sublayer(Norm(x))     # residual stream is NEVER normalized
post-norm (GPT-1):   x = Norm(x + Sublayer(x))      # residual stream IS normalized each layer
```

Pre-norm keeps a clean, un-normalized "residual highway" running the depth of the network, so
gradients flow to early layers unattenuated (Xiong et al. '20, "On Layer Normalization in the
Transformer Architecture"). Post-norm normalizes the highway at every step, which historically
makes deep transformers hard to train without a careful warmup.

### What we measured — and why "stagnation" is the right word
Post-norm finished at **6.8810** (ppl 973) — catastrophic. But the *shape* of the failure is the
lesson, and it's **not** the loss-spike/NaN blow-up most people picture when they hear "unstable":

- `grad_norm` never exceeded **1.52** the entire run (grad_clip=1.0 mostly held it) — **no
  explosion**.
- train_loss dropped fast to ~6.8 by **step ~150**, then flatlined for the remaining **1350 steps**.
- generated samples at step 1400 are degenerate punctuation/word-fragment soup:
  `"not his,, that,, of in of,. the to, of by all;,,..."`

So post-norm didn't blow up — it **got stuck**. The model reached a shallow local optimum around
loss 6.8 (only a bit better than the ~9.5 init) and couldn't escape.

### Why stagnation, mechanistically
This connects directly to a piece of our own model code, `GPT._scale_residual_projections()`. Our
init scales every residual-writing projection by **1/√(2·n_layers)** (GPT-2 §2.3) so that the sum
of 2n roughly-independent sublayer contributions keeps ~constant variance down the depth — a trick
**designed for the pre-norm residual highway**. Under post-norm, the residual stream is
re-normalized every layer, so that carefully-tuned variance budget is disrupted, and the gradient
signal reaching early layers is diluted/noised enough that they stop learning useful updates. The
network's effective depth collapses. Xiong et al.'s pre-norm analysis is exactly this: pre-norm's
un-normalized highway is what lets a 15-layer stack train *at all* at a fixed lr/warmup without
babysitting.

**Why this is a "negative result on purpose":** the phase-5 spec explicitly wants post-norm as a
control at *fixed compute*, not a best-effort post-norm. A fair post-norm would need a much gentler,
longer warmup and probably a lower lr — which would be changing two variables. We measured what
post-norm does with *the baseline's* recipe, and the answer is "it can't use it." That's the point.

---

## 5. QK-norm: the surprise win, and how we know it's real

### The mechanism
QK-norm normalizes the query and key vectors **per head, before the attention score**:

```
scores = (Norm(q) · Norm(k)ᵀ) / √head_dim      # vs plain q·kᵀ / √head_dim
```

Its usual justification (Gemma2, Qwen2) is *stabilizing the scale of attention logits*: without it,
q·k can grow large and push softmax into saturated, low-gradient regions, especially deep/at scale.
The going-in expectation was therefore that QK-norm would be a **no-op or marginal** at our tiny
S-tier/15-layer scale — a "matters at scale" technique.

### What we measured
QK-norm finished **3.4414 vs 3.5037** — **−0.0622**, ~4× the noise floor, and the **best result of
the wave**. The param cost is nil: two RMSNorm(head_dim=64) per attention layer = 15 × 2 × 64 =
**1,920 params** (+0.02%; 9,715,392 vs 9,713,472).

### Why we trust it's real (not a lucky init)
The single most convincing signal is the **shape of the gap over training**, not the endpoint:

| step | 100 | 400 | 800 | 1200 | 1400 |
|---|---|---|---|---|---|
| Δ vs baseline | −0.038 | −0.058 | −0.060 | −0.062 | −0.062 |

The advantage **appears early and widens, then holds** — the opposite of a lucky-init effect, which
would show up as an early bump that *decays* as the baseline catches up. A widening-then-stable gap
is the fingerprint of a genuine optimization-quality improvement: more stable attention logits let
the rest of the network train more effectively throughout, not just at the start. (Caveat, stated
honestly: this is one seed. A −0.062 gap is comfortably past the floor, but a 3-seed repeat would
firm up the *magnitude*. We didn't run it because the *direction* is unambiguous and the recipe
decision — "add it" — doesn't hinge on the exact number.)

**Recipe implication:** carry `qk_norm=true` forward as a recommended default. This is a real
*update* to D-016's baseline, not a confirmation of an existing choice — the one place Wave A tells
us to *change* the default.

---

## 6. The methodological payoff: the noise floor earned its keep on day one

D-035 established the seed-noise floor (0.0150) precisely so verdicts could separate "real" from
"noise." Wave A was its first real use, and it mattered in two directions:

- **It demoted a fake win.** LayerNorm's −0.0158 *looks* like a win at face value. Against the
  floor, it's a wash — the honest call is "keep RMSNorm for cost," not "switch to LayerNorm." Without
  the floor we'd have logged a spurious result.
- **It certified the real ones.** SwiGLU (+0.173 = 11× floor) and QK-norm (−0.062 = 4× floor,
  widening) are *comfortably* past the floor — we can state them as real without a 3-seed repeat.
  Post-norm (225× floor) needs no discussion.

The general lesson, reusable for every later wave: **always quote the delta as a multiple of the
noise floor, and read the whole trajectory, not just the endpoint.** A 1× delta is a coin flip; a
4×+ widening delta is a finding.

---

## 7. What Wave A does *not* establish (and when to revisit)

- **S-tier only.** 10M params, 98.3M tokens, seq 512, one seed each. The QK-norm win and the
  SwiGLU>GELU gap are *expected* to hold or grow at M/L scale (both are usually framed as
  scale-friendly), but "expected" isn't "measured." Before locking the phase-9 recipe, spot-check
  QK-norm at the capstone tier.
- **RMSNorm≈LayerNorm is a single-seed near-tie.** If it ever mattered (it won't — we keep RMSNorm
  regardless), it'd need 3 seeds to resolve.
- **Post-norm was deliberately given the baseline's recipe.** It says nothing about whether a
  *properly warmed-up* post-norm could compete — that's out of scope (would change two variables).
- **These are not new rework rows.** All revisit items live in D-036's "Revisit if"; per the
  discussion-session rule this note logs no new decisions or RW rows.

---

## 8. Connections (where Wave A plugs into the rest of the project)

- **Wave B (positional):** ran the same way — single-variable swaps off the true 3-head baseline,
  same noise floor. Between them, Waves A and B are the "config-only" waves; the baseline recipe
  they refine (rmsnorm/pre-norm/swiglu **+ qk_norm**) is the starting point every later wave builds
  on.
- **Wave C (attention):** QK-norm is **orthogonal** to attention type — it normalizes q/k before the
  score and would stack on top of GQA/MLA. Our MLA implementation doesn't currently apply qk_norm to
  its assembled q/k; if MLA is chosen for the capstone, "qk_norm + MLA" is an explicit combination
  worth testing (also flagged in Wave C's note).
- **The residual-init connection:** post-norm's stagnation is the clearest demonstration in the
  project of *why* `_scale_residual_projections()`'s 1/√(2n) factor and pre-norm go together — good
  to remember when reasoning about training stability at the L-tier capstone's greater depth.
- **Phase 9 recipe:** Wave A's net contribution to `docs/results/recipe.md` (written once waves A–D
  land) is: **keep rmsnorm + pre-norm + swiglu, add qk_norm; do not switch to layernorm/gelu/
  post-norm.**
- **Papers:** Zhang & Sennrich '19 (RMSNorm), Ba '16 (LayerNorm), Shazeer '20 (SwiGLU/GLU variants),
  Xiong et al. '20 (pre-norm depth stability), Gemma2/Qwen2 tech reports (QK-norm).

---

## 9. Learning checkpoints (answered)

1. **Why does RMSNorm match LayerNorm despite doing "less"?** Because for transformers the
   *re-scaling* (dividing by RMS) is what controls gradient scale through depth; the *re-centering*
   (mean subtraction) LayerNorm adds turns out to be nearly irrelevant. We measured a −0.0158
   difference — right at the noise floor — confirming "matches, not beats." RMSNorm saves the bias
   (5,952 params here) and the centering compute.
2. **Why is SwiGLU's ffn_mult 8/3 while GELU's is 4?** To match matrix params. SwiGLU's 3 matrices
   at hidden = (8/3)·d and GELU's 2 matrices at hidden = 4·d both equal 8·d² params/layer (294,912
   at d=192). SwiGLU then wins by +0.17 val loss at that matched budget.
3. **Post-norm failed — did it blow up or stagnate, and why?** Stagnate. grad_norm stayed ≤1.52
   (no explosion); loss froze near 6.8 by step 150. Post-norm normalizes the residual highway every
   layer, disrupting the variance budget that pre-norm + 1/√(2n) init rely on, so gradients to early
   layers are too diluted to keep learning — effective depth collapses.
4. **Why do we believe QK-norm's win is real and not lucky init?** The gap appears early and
   *widens then holds* (−0.038 → −0.062), the signature of a genuine optimization improvement rather
   than an init fluke (which would decay). It's 4× the noise floor at +0.02% param cost.

---

## 10. Key takeaways to carry forward

1. **Un-bundling the modern stack, three of four defaults are confirmed and one is added.**
   RMSNorm (cost, not quality), pre-norm (essential), SwiGLU (real +0.17 win) — all kept; **QK-norm
   promoted from "off" to recommended default** (real −0.062 win, widening).
2. **The failures teach as much as the wins.** Post-norm's *stagnation* (not divergence) is the
   project's cleanest illustration of why pre-norm + scaled residual init are a package.
3. **The noise floor immediately paid off** — it demoted LayerNorm's face-value "win" to a wash and
   certified SwiGLU/QK-norm as real. Always report deltas in units of the floor, and read the whole
   trajectory.
4. **Param-matching is arithmetic you can and should verify** (8·d² both ways; the 14.4K GELU bias
   asterisk). Matching "params" loosely hides confounds; matching them exactly and naming the
   residual difference is the honest way.
5. **Config-only waves are cheap and high-yield** — 4 real findings, zero new code, ~1 hr of GPU.
   Spend the expensive sessions (Wave C's MLA) where implementation is actually required.

**Open questions for later:** does QK-norm's win hold/grow at M/L tier? Does qk_norm stack cleanly
with MLA? Both deferred to the phase-9 capstone recipe (D-036, and cross-referenced in D-038).
