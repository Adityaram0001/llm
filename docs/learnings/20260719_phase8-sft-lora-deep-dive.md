# Phase 8 Parts A & B deep dive — SFT and LoRA from scratch

**Session type:** discussion (no code/specs changed). **Date:** 2026-07-19.
**Companion records:** D-051 (SFT), D-052 (LoRA), `docs/results/finetune_report.md`, runs
`experiments/20260719_p8_sft-s-dictionary[-lora-*]`. This note is the "revise later" version — it
re-derives the mechanics and answers the doubts raised in the session, with the project's real
numbers. If you only reread one phase-8 file before the capstone, read this one.

---

## 0. The one-paragraph story

We took the S-tier pretrained base (`p4_s_baseline`, 9.71M params, a plain next-token language
model) and turned it into a model that *answers questions and stops*, using 2,916 dictionary Q&A
pairs from the phase-7 factory. Part A did it by **full fine-tuning** (update every weight); Part B
did it with **LoRA** (freeze the base, train tiny low-rank adapters). The headline: at this scale
the *behavior* changed dramatically (a base LM that rambles → a model that answers-and-stops), the
*knowledge* barely changed (a 10M model has little to surface), and LoRA matched or beat full FT
while training ~2% of the parameters. Everything below unpacks why.

---

## 1. What "ppl" (perplexity) actually means

Every quality number in this project is ultimately **cross-entropy loss**, and perplexity is just a
friendlier way to read it.

**Cross-entropy (CE).** For each token position the model outputs a probability distribution over
the 16,000-token vocabulary. CE is the average, over all positions, of `−ln(probability the model
assigned to the token that actually came next)`. If the model is confident and right, that
probability is near 1 and `−ln(1)=0`; if it's wrong/uncertain, the probability is small and
`−ln(small)` is large. So **lower CE = better predictions**. CE is measured in *nats* (natural-log
units).

**Perplexity = exp(CE).** It re-expresses the loss as an "effective number of equally-likely
choices" the model is torn between at each step:

| situation | CE (nats) | perplexity | reading |
|---|---:|---:|---|
| perfect model (prob 1 on truth) | 0 | 1 | never surprised |
| our base on held-out books+dict | **3.553** | **34.9** | as unsure as a fair 35-way guess |
| uniform guess over the whole vocab | ln(16000)=9.68 | 16000 | knows nothing |

So `ppl = 34.9` means: out of 16,000 possible next tokens, the model has narrowed its genuine
uncertainty down to *about 35 equally-plausible options per token* — far better than chance (16000),
far from perfect (1). Worked both directions: `exp(3.553)=34.9` and `ln(34.9)=3.553`.

**Bits, for intuition.** Divide nats by `ln(2)`: `3.553 / 0.693 = 5.13 bits/token`. The model needs
~5.1 yes/no questions to pin down each token. (The eval suite also reports *bits-per-byte*, which
normalizes by raw text bytes so it's comparable across tokenizers — see the phase-6 note.)

**Why we care here:** perplexity is the ruler we hold up to the frozen pretrain data to *see*
forgetting (next section), and `exp(SFT val loss 3.828) = 46.0` is the perplexity of the fine-tuned
model on held-out dictionary answers.

---

## 2. Catastrophic forgetting: what "forgetting +25%" means and how it's measured

**The phenomenon.** When you fine-tune a model on a narrow new task, it gets better at that task but
*worse* at everything it used to know — its weights move toward the new distribution and away from
the old one. This is "catastrophic forgetting." It's not a bug; it's the default behavior of
gradient descent on a new objective.

**How we measure it (the exact mechanic).** `SFTTrainer` keeps a **frozen probe**: at construction
it samples 16 fixed batches from the *pretrain* validation set (`data/tokenized/hf_bpe_16k/val.bin`
— held-out books + dictionary tokens, the same data the base was evaluated on). At every SFT eval
step it runs the *current* model over those same 16 batches and computes **plain next-token
cross-entropy** — not the masked SFT loss, the ordinary language-modeling loss — via
`last_aux_metrics["ce_loss"]` (the pure-CE field the model exposes, added back in Wave F). Same
batches every time, so any change is real drift, not sampling noise. We log both the loss and
`pretrain_val_ppl = exp(loss)`.

**"Forgetting +25%" decoded.** It's the *percentage rise in that pretrain perplexity* from the start
of SFT (step 0, before any gradient — this equals the base model's ppl, 34.9) to the end:

| run | pretrain ppl: start → end | rise | in loss (nats) |
|---|---|---:|---:|
| full FT (lr 2e-5) | 34.9 → 40.1 | **+14.8%** | +0.138 |
| LoRA r8 attn (lr 5e-4) | 34.9 → 44.4 | **+27.1%** | +0.240 |
| LoRA r32 attn | 34.9 → 43.4 | +24.2% | +0.216 |
| LoRA r8 attn+ffn | 34.9 → 48.0 | +37.3% | +0.303 |

`+27.1%` = `44.4 / 34.9 − 1`. Equivalently the loss rose `ln(44.4) − ln(34.9) = 3.793 − 3.553 =
0.240 nats`. So the model that used to be "35-way unsure" on books is now "~44-way unsure" — it has
measurably drifted away from general English toward dictionary-answer style.

**The lesson we watched live.** SFT val loss *fell* (5.54 → 3.83) while pretrain ppl *rose* (34.9 →
40.1). That divergence — getting better at the target while getting worse at the origin — **is**
catastrophic forgetting, made visible in two curves. The main guard is a **low learning rate**
(full FT used 2e-5; a higher LR or more epochs widens the gap). Other mitigations, not done this
session: mix a slice of pretrain data into the SFT stream, use fewer epochs, or use LoRA (see §6 for
the important reversibility nuance).

---

## 3. The SFT mechanic: chat template + assistant-only loss masking

**Why a template at all.** A pretrained LM only *continues text*. It has no concept of "a turn" or
"who is speaking." Before SFT, asked "What does ephemeral mean?", our base just continued in
book-prose and never stopped (stop-rate 0%). SFT teaches a fixed *protocol* using tokens reserved
back in phase 2 (D-014) so their IDs never shifted: `<|user|>`=2, `<|assistant|>`=3,
`<|endoftext|>`=0, `<|pad|>`=1. One training example renders as:

```
<|user|>What does methylic mean?<|assistant|>Methylic describes ... methyl.<|endoftext|>
```

At inference we feed everything up to and including `<|assistant|>` and let the model generate; it
learns to produce the answer and then emit `<|endoftext|>` to stop. This is exactly the chat-ML idea
behind the messages APIs you already know — just spelled out at the token level.

**Loss masking — the single most important mechanic.** We do NOT want the model trained to generate
user turns (it would learn to *ask* questions). We only want it to learn the *assistant* content and
the stop token. So we build a per-token `supervise` mask and set every non-supervised target to the
ignore index `−1`, which the model's cross-entropy already honors (`ignore_index=-1`, present since
phase 3). Worked example (schematic token IDs):

```
token:      <|user|> What does methylic mean? <|assistant|> Methylic describes ... methyl <|endoftext|>
supervise:      0     0    0     0      0            0           1        1     ...   1          1
```

Then the standard next-token shift: `x = tokens[:-1]`, `y = tokens[1:]`, and `y` is set to `−1`
wherever `supervise (shifted) == 0`. Position *i* of the model predicts token *i+1*; loss counts
only where the predicted token is assistant content or the ending `<|endoftext|>`. Concretely, in a
padded batch most target positions are `−1` (prompt tokens + right-padding) and only the ~15–30
answer tokens per example drive the gradient.

**One subtlety we got right by construction.** We tokenize each turn's *content* separately (with
`add_special_tokens=False`) and splice in the known marker IDs, rather than tokenizing the whole
string and guessing where the assistant span starts by counting tokens. That guessing is exactly
the BPE-boundary fragility that caused eval bug RW-6. Here the mask is *exact* — a test asserts the
spliced encoding equals a single `encode()` of the full string.

**Padding vs packing.** Our examples are tiny (p99 ≈ 67 content tokens, max 93). We right-pad to the
batch max with `<|pad|>`. Under a causal model this needs no correction: real tokens never attend to
*later* pad tokens (causality), and pad *targets* are `−1` so they carry no loss. Packing
(concatenating examples to fill every window) would save compute, but at these lengths there's
almost nothing to save and it muddies the mask you want to *see*.

---

## 4. Full FT vs the three LoRA configs — technically, and in our experiment

This directly answers "what is the difference between full FT, LoRA r8 attn, LoRA r32 attn, LoRA r8
attn+ffn, technically and with respect to our experiment."

### 4a. First, where the parameters live in one S-tier block

The S model has 15 blocks; each block has **attention** (4 linear projections) and a **SwiGLU FFN**
(3 linear projections). All attention projections are 192×192; FFN uses hidden = 8/3 × 192 = 512.

| projection | shape | full params |
|---|---|---:|
| attn q/k/v/o (×4) | 192×192 each | 36,864 each |
| ffn gate/up | 512×192 each | 98,304 each |
| ffn down | 192×512 | 98,304 |

Across the whole model: attention ≈ 2.21M params, **FFN ≈ 4.42M params** (≈ 2/3 of the
non-embedding budget), embeddings 3.07M (tied), norms tiny. **Remember this 1/3-attn vs 2/3-ffn
split — it explains the placement result.**

### 4b. What each method trains

- **Full FT** — *every* one of the 9,713,472 params is trainable. AdamW keeps a gradient + two fp32
  moments per param. The base is overwritten (destroyed): there is no "original" to go back to.
- **LoRA r8 attn** — freeze everything, then wrap the **60 attention projections** (4 per block ×
  15) each with a rank-8 adapter. Only the adapters train: **184,320 params (1.9%)**. The FFN,
  embeddings, norms, and head never move.
- **LoRA r32 attn** — same 60 layers wrapped, but rank 32 (4× more directions per adapter):
  **737,280 params (7.6%)**.
- **LoRA r8 attn+ffn** — wrap attention **and** the 3 FFN projections per block → **105 layers**,
  rank 8: **437,760 params (4.5%)**. This is the only config that lets the FFN (the 2/3 majority of
  the compute) adapt.

(Adapter param math per layer, r=8, a 192×192 linear: `A` is 8×192=1536, `B` is 192×8=1536, total
3072 — vs 36,864 full, a 12× reduction. `60 × 3072 = 184,320`. ✓)

### 4c. What actually happened in our experiment (same base, same data, 3 epochs, seed 1337)

| method | trainable | AdamW opt-state | ckpt (deliverable) | best SFT val | stop-rate |
|---|---:|---:|---:|---:|---:|
| full FT (lr 2e-5) | 9.71M (100%) | 116.6 MB | 116.7 MB | 3.828 | 80% |
| LoRA r8 attn (lr 5e-4) | 0.18M (1.9%) | 2.2 MB | **0.77 MB** | 3.942 | 95% |
| LoRA r32 attn | 0.74M (7.6%) | 8.9 MB | 2.98 MB | 3.924 | 95% |
| LoRA r8 attn+ffn | 0.44M (4.5%) | 5.3 MB | 1.80 MB | **3.777** | 90% |

Three readings:

1. **Placement beats rank.** Going r8 → r32 (4× the rank, attention only) barely moved val loss
   (3.942 → 3.924 — a difference smaller than the pretraining noise floor of 0.015, so effectively a
   tie). But *adding the FFN* at the same rank 8 (attn+ffn) jumped to **3.777, beating full FT**.
   Why: attention-only LoRA can only influence ~1/3 of the model's computation; the FFN holds 2/3 of
   the params and is where much of the token-shaping work happens. No amount of extra rank on
   attention compensates for leaving the FFN frozen. **Where you put the adapters matters more than
   how big they are** — at least on this task.
2. **LoRA is competitive-to-better, not a compromise.** The best LoRA (3.777) *beat* full FT
   (3.828), and every LoRA variant answered-and-stopped more reliably (90–95% vs 80%). Plausible
   reasons: (a) the low-rank constraint regularizes against overfitting a tiny 2,916-example set;
   (b) LoRA can use an aggressive LR (5e-4) because the frozen base anchors general competence, so it
   sidesteps the learn-vs-forget tradeoff that forced full FT to a timid 2e-5; (c) full FT at
   2e-5/3-epochs may simply be a bit under-tuned.
3. **The memory win is the point (see §5).**

---

## 5. LoRA math from scratch: W + (α/r)·BA, why B=0, and where the savings come from

**The decomposition.** A fine-tune wants to learn an *update* ΔW to a weight `W` (h = (W + ΔW)x).
LoRA's bet: the useful ΔW for a downstream task is approximately *low-rank*, so factor it as
`ΔW = B A` with `A: (r, in)`, `B: (out, r)`, `r ≪ min(in, out)`. Then

```
h = W x  +  (α / r) · B A x
```

`W` is frozen; only `A` and `B` train. A full update has `out × in` free parameters; the low-rank
one has `r × (out + in)` — for a 192×192 layer at r=8 that's 3,072 vs 36,864 (**12×** fewer).

**Why B is initialized to zero (and A random).** At step 0, `B = 0` ⟹ the adapter output
`(α/r)·B·A·x = 0`, so the adapted model is **bit-identical to the pretrained model**. Fine-tuning
starts from the base's behavior, not from a random perturbation of it — critical, because a random
ΔW at init would corrupt a good model before training even begins. Gradients still flow: `dL/dB ∝
(dL/dh)·(A x)ᵀ`, which is nonzero because `A` is random, so **B moves on step 1**; once `B ≠ 0`,
`A` moves too. If *both* were zero, `dL/dA ∝ Bᵀ(dL/dh)` would also be zero and the adapter would be
permanently dead — so exactly one of the pair must be nonzero at init. (Tests assert both: adapter
output == base output at init, and gradient reaches `B`.)

**The rank ↔ alpha relationship (a doubt raised explicitly).** Two independent knobs:
- **r (rank)** = *how many* directions the update can move in — the expressiveness/capacity of ΔW.
- **α (alpha)** = a fixed scalar that sets *how strongly* the adapter's output is added, via the
  scaling factor `α/r`.

The `α/r` form is deliberate: it **decouples update strength from rank**. If you scale α *with* r
(e.g. α = r, or the common α = 2r), then `α/r` stays constant, so increasing rank adds capacity
*without* also cranking the update magnitude — the two knobs stay independent, and you can raise r
without re-tuning the learning rate. Think of it as: **r = how many dials, α/r = how far each dial
is allowed to turn.**

⚠️ **A caveat about our own rank sweep.** We held **α = 16 fixed** across r8 and r32. So the scaling
was *not* constant: `α/r = 16/8 = 2.0` for r8 vs `16/32 = 0.5` for r32. Our "rank sweep" therefore
also quartered the per-direction scaling — it's a mild confound, not a clean rank-only comparison.
It doesn't change any conclusion (r8≈r32 was a tie either way, and placement dominated), but a
*clean* future rank sweep should hold `α/r` constant (set α ∝ r). Flagged in the parking lot.

**Where the memory savings actually come from — and where they don't.** LoRA does *not* save
forward FLOPs or activation memory: the frozen base still runs in full, and the extra low-rank
matmul is tiny (so LoRA's throughput was 13–16K tok/s, comparable — its win is **memory, not
speed**; attn+ffn is the slowest of the three only because it adapts 105 layers vs 60). The saving
is entirely in the **optimizer + gradient state**. AdamW stores, *per trainable parameter*, a
gradient and two fp32 moments (m, v) ≈ 3 × 4 = 12 bytes. So:

```
full FT   : 9,713,472 trainable × 12 B ≈ 116.6 MB of optimizer/grad state
LoRA r8   :   184,320 trainable × 12 B ≈   2.2 MB   (53× less)
LoRA r32  :   737,280 trainable × 12 B ≈   8.9 MB   (13× less)
```

**Why it's "transformative at 7B+" but merely tidy here.** At 10M params, 116 MB of optimizer state
is nothing — the model trains fine either way. But the *ratio* is scale-invariant: a 7B model needs
~84 GB of AdamW state for full FT (7B × 12 B) — more than a single 80 GB GPU — while a rank-16 LoRA
needs a few hundred MB. That is the difference between "fits on one GPU" and "doesn't," which is
exactly why LoRA is standard for large-model fine-tuning (and why you could fine-tune Gemma-12B on
modest hardware). The checkpoint story mirrors it: our LoRA *deliverable* is the 0.77–2.98 MB
adapter, vs a ~39 MB dense model (or the 116.7 MB `best.pt` that also carries optimizer state).

**Merge-back.** `merge_lora` folds each adapter into its base weight (`W ← W + (α/r)·B A`), turning
the model back into a plain dense `nn.Linear` stack — same outputs (a test asserts it), zero adapter
overhead at inference. So you get LoRA's training-time memory win *and* a normal model to serve.

---

## 6. The forgetting reversibility nuance (why the +% table is not the whole story)

Reading §2's table naively says "LoRA forgot *more* (+24–37%) than full FT (+15%) — so LoRA is worse
for forgetting." That reading is wrong in two ways:

1. **It's LR-confounded.** Our LoRA runs used lr 5e-4 — 25× the full-FT 2e-5. A larger LR moves the
   model further in output space (on *both* the SFT task and the pretrain probe), so more measured
   drift is expected. An apples-to-apples, LR-matched comparison hasn't been run (flagged).
2. **LoRA's forgetting is fully *reversible*; full FT's is permanent.** The pretrain-ppl rise we
   measure is on the *adapted* model (base + adapter). But the base weights are **frozen and
   bit-identical** to `p4_s_baseline`. Pop the adapter off (or just don't apply it) and pretrain ppl
   is *exactly* 34.9 again — provably, because the base checkpoint is loaded unchanged. Full FT
   overwrites the weights: there is no original to recover.

So the true LoRA forgetting-safety property is **reversibility / non-destruction of the base**, not
"lower adapted-model perplexity." You can keep one frozen base and swap in different task adapters
without ever degrading the base — a property full FT structurally cannot offer.

---

## 7. Two bugs the fp32/CPU tests missed — and why "run it" caught them

Both LoRA bugs passed every unit test (which run fp32 on CPU) and only surfaced on the first *real*
MPS + bf16 run — a concrete instance of CLAUDE.md's "verify on a real run" rule:

1. **bf16 autocast dtype clash.** Under bf16 autocast the frozen base outputs bf16, but the raw
   `x @ lora_A.t()` is *not* an autocast-covered op on MPS, so it kept the fp32 adapter params →
   `RuntimeError: BFloat16 != float`. Fix: do the adapter matmuls with `F.linear`, which *is*
   autocast-eligible and casts params exactly like `nn.Linear`. (Lesson: hand-rolled `@` on
   parameters bypasses autocast's casting; prefer `F.linear`.)
2. **Device placement.** `apply_lora` creates the new `A`/`B` params with `torch.empty(...)`, which
   defaults to CPU — while the model was already on MPS → `weight is on cpu but expected on mps`.
   Fix: `model.to(device)` *after* wrapping, in both `SFTTrainer` and `load_finetuned`.

Both crashed after `__init__` but before the first eval, leaving adapter-only `latest.pt` and no
metrics — and each crash still appended a registry row via `fit()`'s `finally`. Those 8 stale rows
were removed by hand so the lab record stays honest (same discipline as D-046's spurious-row fix).
**Takeaway: green unit tests are necessary, not sufficient — dtype/device bugs live specifically in
the paths tests don't exercise (autocast, the real accelerator).**

---

## 8. Answers to the session's explicit doubts (index)

- *What is ppl?* → §1. Perplexity = exp(cross-entropy); "effective number of equally-likely next
  tokens." Base = 34.9 out of a 16,000 vocab.
- *What does "forgetting 25%" mean and how is it measured?* → §2. The % rise in **pretrain-set
  perplexity** (plain LM loss on frozen held-out books+dict batches) from step 0 to end;
  `44.4/34.9 − 1 = +27.1%` = the model drifting away from general English.
- *Full FT vs LoRA r8 attn vs r32 attn vs r8 attn+ffn — technically and in our experiment?* → §4.
  Which/how-many matrices train, exact param counts, and the result table (placement > rank; best
  LoRA beat full FT).
- *Rank ↔ alpha relationship?* → §5. r = number of directions (capacity); α/r = update strength;
  the α/r form decouples them; our sweep held α fixed so scaling wasn't constant (caveat).

---

## 9. Takeaways (revision checklist)

1. **Perplexity is just `exp(loss)`** — an uncertainty ruler; 1 = perfect, vocab-size = clueless.
2. **SFT teaches behavior via a loss mask** — only assistant/stop tokens carry loss; everything else
   is `−1` (ignored). Build the mask by construction, don't infer boundaries from token counts.
3. **Forgetting is the SFT-loss-down / pretrain-ppl-up divergence**, measured on a frozen probe;
   low LR is the main guard.
4. **LoRA freezes W and learns (α/r)·BA**; B=0 at init ⟹ starts as the base; r = capacity, α/r =
   strength.
5. **LoRA's win is optimizer memory (13–53× here, and the *ratio* is what matters at 7B+), not
   speed**; the adapter is a KB-to-MB deliverable, and its forgetting is *reversible*.
6. **Placement can beat rank** — adapt the FFN (2/3 of the compute), not just attention.
7. **Green CPU/fp32 tests miss dtype/device bugs**; run on the real accelerator before trusting.

## 10. Open questions / flagged follow-ups (→ parking lot)

- **Clean rank sweep:** hold `α/r` constant (α ∝ r) so r8-vs-r32 isolates rank, not scaling.
- **LR-matched forgetting:** rerun full FT and LoRA at the *same* LR to compare adapted-model
  forgetting fairly (the reversibility conclusion stands regardless).
- **Multi-base SFT addendum** (user-requested, post-Part-C): does better pretrain loss predict a
  better SFT model? Cheapest clean set = `p4_s_baseline` / `wave-d-muon` / `wave-d-wsd` (same
  architecture, drop-in loadable).

## 11. Links
- Decisions: D-051 (SFT), D-052 (LoRA); eval bug context RW-6 (§3), Wave F ce_loss field (§2).
- Runs: `20260719_p8_sft-s-dictionary`, `…-lora-r8-attn`, `…-lora-r32-attn`, `…-lora-r8-attnffn`.
- Report/data: `docs/results/finetune_report.md`, `docs/results/finetune_partB.json`.
- Papers: LoRA (Hu et al. '21); weight tying / why we skip `lm_head` (Press & Wolf '16, D-016);
  chat-ML protocol (connects to the messages-API mental model).
