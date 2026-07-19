# Phase 8 Part C deep dive — DPO from scratch

**Session type:** discussion (no code/specs changed). **Date:** 2026-07-19.
**Companion records:** D-053, `docs/results/finetune_report.md` Part C, run
`experiments/20260719_p8_dpo-s-dictionary` (`notes.md`, `metrics.jsonl`, `eval_dpo.json`),
`notebooks/09_dpo_from_scratch.ipynb`. This note goes past what notes.md/D-053 captured — in
particular §3 runs a **new analysis** (a per-failure-mode reward breakdown) that wasn't done
during the training session and meaningfully sharpens the "length confound" story. If you only
reread one phase-8 file before the capstone, read the SFT/LoRA sibling
(`20260719_phase8-sft-lora-deep-dive.md`) for the mechanics and this one for what happens when
fine-tuning becomes *contrastive*.

---

## 0. The one-paragraph story

DPO took the SFT model (Part A) and refined it further using 2,884 preference pairs: `chosen` =
the already-good phase-7 SFT answers, `rejected` = a deliberately-corrupted answer in one of
three styles (wrong_fact / off_format / verbose). Training was real and fast — by less than half
an epoch the model discriminated chosen from rejected 95.8% of the time on held-out pairs — but a
**new breakdown by failure mode** (§3) shows that "95.8%" hides an uneven story: the model learned
to detect *bulk* (off_format, verbose — both much longer than chosen) almost perfectly and fast,
but *factual correctness* (wrong_fact — the same length as chosen) only 87.2% of the time, with a
much weaker gradient signal. The answer-length collapse (34→16.5 tokens) is the visible symptom of
that imbalance, not a mysterious side effect. The run was also stopped early on a real,
un-explained MPS slowdown — an honest "here's what we know and don't" story, not swept aside.

---

## 1. From RLHF to DPO — the derivation, and the connection you already know

The full symbol-by-symbol derivation is in the notebook (§§1-5); here's the same four steps with
the connections to statistics you already have from sklearn/logistic regression.

**Step 1 — the objective.** RLHF wants a policy that scores well under a reward `r(x,y)` but
doesn't wander too far from a reference policy `π_ref` (the SFT model): maximize
`E[r(x,y)] − β·KL(π_θ‖π_ref)`. Read `β` exactly like a **regularization strength** — bigger `β`
means the reward term dominates and the policy is allowed to move further from the reference;
smaller `β` clamps it closer, the same role a bigger `λ` plays in ridge regression, just written
as a Lagrange multiplier on a constraint rather than a penalty on weights.

**Step 2 — for a FIXED reward, the optimal policy has a closed form.** This is a standard
result: the KL-regularized-reward-maximizing distribution is a **Boltzmann/Gibbs reweighting** of
the reference — `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`. This is the same math behind a softmax
policy with an entropy/KL bonus in classic RL, or a Bayesian posterior that reweights a prior by a
likelihood — "start from what you already believed, tilt it exponentially toward what scores
well." The intractable part is `Z(x)`, the normalizer summing `exp(r/β)` over every possible
response — nobody computes this directly.

**Step 3 — invert it: reward in terms of policy.** Solve Step 2 for `r`:
`r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)`. The `β·log Z(x)` term depends on `x` only, not
`y` — remember that; it's about to vanish for free.

**Step 4 — Bradley-Terry, and the killer connection.** Human/synthetic preference data doesn't
give you `r(x,y)` directly, only *which of two responses won*. The standard model for that
(Bradley & Terry, 1952 — originally for ranking chess/sports players from pairwise results) is
`P(y_w ≻ y_l | x) = σ(r(x,y_w) − r(x,y_l))`. Substitute Step 3's expression for `r` on both sides:
the `β·log Z(x)` term is IDENTICAL for `y_w` and `y_l` (same `x`), so it **cancels in the
subtraction**. What's left:

```
P(y_w ≻ y_l | x) = σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) )
```

**This is a logistic regression.** Not "like" one — literally one. In sklearn terms:
`P(label=1) = σ(w·features)`. Here the "features" are the two log-ratio terms (each a scalar the
model computes via a forward pass, not a hand-engineered feature), the "weight" is the fixed
scalar `β`, and the "label" is which response a human/synthetic annotator preferred. `dpo_loss` in
`src/llmlab/train/dpo.py` is `-F.logsigmoid(reward_chosen - reward_rejected).mean()` — that IS
sklearn's `log_loss`/binary cross-entropy, just with model-computed log-ratios standing in for
`X @ w`. **This is DPO's entire trick**: an RL problem with an intractable normalizer, reduced
algebraically to "fit a logistic regression whose features the model supplies about itself." No
reward model is ever trained; `r` was a symbol used to get here and then eliminated (Rafailov,
Sharma, Mitchell, Ermon, Manning, Finn — [*Direct Preference Optimization: Your Language Model is
Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), NeurIPS 2023).

---

## 2. What the reward margin actually says, in nats and in odds (worked)

At step 75 (the checkpoint actually evaluated), val `reward_margin = 8.9694` at `beta = 0.1`.

`reward_margin = β · (Δ_chosen − Δ_rejected)` where each `Δ = logπ_θ(y|x) − logπ_ref(y|x)`. So the
**average combined log-ratio gap** is `margin / β = 8.9694 / 0.1 = 89.694 nats`. Exponentiate to
read it as an odds ratio: `e^89.694 ≈ 1.2 × 10^39`. That number has no realistic interpretation as
"how much more likely" — it's a sign the policy and reference have diverged into regions where the
Bradley-Terry probability is saturated at ≈1 for nearly every pair, i.e. the logistic regression
analogy from §1 has hit the flat part of the sigmoid where gradients are tiny but the *underlying*
log-ratio can still grow unboundedly. Compare to **step 0**, where policy ≡ reference exactly, so
every log-ratio is 0, `margin = 0`, and loss `= -logσ(0) = log 2 ≈ 0.693` (verified in the
notebook's toy-number checks and in the real run's own step-0 log). Going from a margin of 0 to a
margin representing an implied `10^39` odds ratio in **75 steps, under half an epoch**, is the
quantitative face of "this saturated fast" — not a vague impression from watching numbers plateau.

---

## 3. The length confound — full mechanism, real arithmetic, and a new breakdown that refines the story

### 3a. Why a raw (non-reference) log-prob comparison is confounded by length

`sequence_logprobs` sums token log-probs over the assistant span:
`Σ_t logπ(token_t)`. Every term is `≤ 0` (it's a log-probability). If a model's average per-token
log-prob over some stretch of text is roughly `c` (a rough approximation, not exact — see 3c), the
total for an `n`-token response is roughly `c·n`. **More tokens ⇒ more negative total, almost
mechanically, regardless of content quality.** This is why the eval script's *reference-free*
check (`raw_logprob_preference`, added in this session after the training run to answer "did SFT
already know this?" more honestly than the degenerate self-vs-self reward_accuracy) needs a big
caveat: it found the SFT model "already prefers chosen 95.8% of the time" with a **+623.7 nat**
average gap — which sounds like SFT already understood the preference, but chosen averages 38.0
tokens on the val set and rejected averages 162.8 — a **124.9-token difference**. `623.7 / 124.9 ≈
5.0 nats per extra token` — a plausible per-token cost, meaning the length difference alone could
explain most of that gap without any content understanding at all.

### 3b. Why DPO's OWN reward is *mostly*, not *perfectly*, immune — a subtlety worth getting right

The reward-vs-reference formulation, `β(logπ_θ(y|x) − logπ_ref(y|x))`, compares the **same
response `y`** under two models — same tokens, same length `n` — so the length-scaled floor
`c·n` cancels **to the extent `c` is the same for both models**. That's the important
qualifier: if training has shifted the policy's *average* per-token confidence uniformly (not
impossible after training on a narrow, small dataset), `c_θ ≠ c_ref`, and a residual length
dependence survives. The reward diagnostic is far more robust to length than the raw comparison —
but "reference-relative" is not a magic length-invariance proof, it's a strong mitigation. This is
exactly the subtlety later DPO literature engages with directly rather than assuming away (see §3d).

### 3c. New analysis this session: reward broken down by failure mode (not just averaged)

The original eval only reported the *aggregate* reward_accuracy/margin (95.8%, 8.9694). Breaking
it down by the three failure modes on the same 144 held-out val pairs — a query the training
session never ran — tells a much sharper story:

| failure mode | n (val) | rejected length vs chosen | reward_accuracy | reward_margin |
|---|---:|---:|---:|---:|
| `wrong_fact` | 47 | 1.38× (53.1 vs 38.6 tok, train-set means) | **87.2%** | **1.690** |
| `off_format` | 51 | 2.56× (98.8 tok) | 100.0% | 5.639 |
| `verbose` | 46 | 9.55× (368.5 tok) | 100.0% | 20.100 |

(Weighted check: `(47×1.690 + 51×5.639 + 46×20.100) / 144 = 1291.6 / 144 = 8.97` — reproduces the
reported aggregate margin exactly, confirming the breakdown is internally consistent.)

**Two real conclusions, both true at once:**

1. **Genuine content learning did happen.** `wrong_fact` pairs are only 1.38× the length of
   chosen — barely a length signal at all — and the model still separated them correctly 87.2% of
   the time (well above the ~50% a pure length-blind guesser would get, and this is a strictly
   *harder* discrimination than the other two modes: a fluent, right-length, wrong-fact answer
   *looks* fine on the surface). This directly refutes reading the length-confound finding as "DPO
   learned nothing real" — it learned something real, just unevenly.
2. **The gradient signal was massively imbalanced toward the two length-correlated modes.**
   `verbose`'s margin (20.1) is **~12× `wrong_fact`'s** (1.69), despite verbose being the *easiest*
   distinction to make (a 10×-too-long answer is trivially detectable by bulk alone) and
   `wrong_fact` being the one the phase actually cares most about getting right (the model
   *knowing* something, not just behaving well). Two of the three failure modes (`off_format` +
   `verbose`, 2/3 of the training data) correlate strongly with length; only one (`wrong_fact`,
   1/3) doesn't. **This — not a vague "DPO exploited length" — is the mechanistic explanation for
   the answer-length collapse** (33.9→16.5 tokens): with 2/3 of the training signal rewarding
   "produce less text" as a valid, easy way to raise the margin, and only 1/3 demanding the harder
   "get the fact right," the average gradient direction tilts toward terseness well before
   fact-checking is mastered — visible directly in `wrong_fact`'s comparatively weak 87.2%/1.69,
   the run's true bottleneck metric.

### 3d. This is documented, real territory — not a bug in this implementation

- **[A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716)**
  (Singhal, Goyal, Xu, Durrett, 2023) shows RLHF reward improvements are often driven mostly by
  response length rather than the intended quality signal. Note the **direction is usually the
  opposite of ours** — typical human-preference data rates longer answers as better (more
  "thorough-looking"), so RLHF models tend to get *more* verbose over training. Ours went the
  other way because *we* built the length/badness correlation backwards on purpose (verbose was
  one of the three deliberately-bad patterns) — same underlying mechanism (the model exploits
  whatever surface feature correlates most cheaply with the label), opposite sign, because the
  training data's own correlation structure has the opposite sign. This is worth sitting with:
  **the confound isn't inherent to DPO, it's inherited from whatever correlates with "bad" in the
  training set** — the ML-general lesson here is identical to a classifier learning a spurious
  leaky feature (a hospital watermark instead of the actual pathology) rather than the intended
  signal, just playing out in preference space instead of classification space.
- **[SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)**
  (Meng, Xia, Chen, NeurIPS 2024) replaces DPO's reward with the **length-normalized average**
  log-prob (`(1/|y|)Σ logπ(y|x)`, not the sum) specifically to remove length as a usable shortcut,
  and reports SimPO-tuned models don't inflate length the way DPO-tuned ones often do. This is the
  most direct, concrete fix for a future re-run here: divide `sequence_logprobs`'s output by
  sequence length before computing the reward (documented follow-up, not done this session).
- **[Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://arxiv.org/abs/2406.02900)**
  (Rafailov, Chittepu, Park, Sikchi, Hejna, Knox, Finn, Niekum, NeurIPS 2024 — same first author as
  the original DPO paper) studies exactly this failure mode across DPO/IPO/SLiC at multiple model
  scales and finds **smaller models show over-optimization almost immediately**, while larger
  models have a more favorable win-rate/KL tradeoff. Our 9.71M-param model saturating within 53% of
  one epoch (§2's `10^39`-odds-ratio number) is a small, clean, in-house reproduction of exactly
  that scale-dependence — a genuinely nice tie-in, not a coincidence: small models have less
  capacity to represent a nuanced, multi-axis preference function, so they collapse onto whichever
  single axis (here: length) gives the cheapest loss reduction.

---

## 4. Why the run was stopped at step 91/172 — what we know, and what we honestly don't

**The timing (from the real run log):** step-block averages were roughly 4.3s/step (steps 0-25),
12.3s/step (25-50), 26.2s/step (50-75), then fluctuating 15-35s/step through step 91 — a steep
early acceleration that seemed to be leveling off, not a clean geometric blowup, by the time it was
interrupted.

**What was checked and RULED OUT:** batch-shape variance. `DPODataset` pads chosen/rejected to
their own per-batch max width, so different batches genuinely do have different compute costs —
but directly inspecting the shuffled batch order's rejected-side widths across the visited step
range (`[0:10]` mean 480.8, `[20:30]` mean 441.0, `[45:55]` mean 487.5, `[70:92]` mean 460.0) shows
**no upward trend** — ruling out "the run just happened to draw longer batches later" as the cause.

**What's plausible but NOT root-caused:** DPO runs **four** forward passes per step (policy ×
{chosen, rejected}, reference × {chosen, rejected}) versus SFT's one short one, on sequences
averaging several hundred tokens (vs SFT's ~30-90) — a genuinely much heavier per-step MPS
workload sustained over minutes, which is consistent with (but not proof of) either M4 thermal
throttling under sustained GPU load, or MPS-backend allocator/kernel-cache growth from running two
live models with continuously varying tensor shapes. Both are plausible, neither was isolated this
session — a real open item (parking lot), not a hand-waved one.

**Why interrupt rather than let it finish:** `tqdm`'s own ETA kept inflating as the run progressed
(the "time remaining" estimate grew even as steps completed), which is the practical, in-the-moment
signal that something was structurally getting slower, not just varying — combined with the
val-signal having already saturated (§2's `10^39` number by step 75), continuing had a bad
cost/benefit: real risk of an unsupervised run crossing into "multi-hour," for information that
was mostly re-confirming an already-large divergence rather than teaching something new.
`DPOTrainer`'s SIGINT handling (same mechanism SFT's trainer has, since `p4` — see the sibling
doc's §"resume" discussion from phase 4) made stopping cleanly a zero-cost decision: `latest.pt`
saved, a real registry row written, nothing lost.

---

## 5. Reading every number in the before/after table

| metric | base | SFT | DPO | what it actually measures |
|---|---:|---:|---:|---|
| stop-rate | 0% | 82% | **99%** | fraction of held-out instructions where the model emits `<\|endoftext\|>` within budget. **Length-unconfounded** — `off_format`'s penalty is specifically for not-answering, independent of how long the eventual non-answer is, so this improvement is a real, clean win. |
| mean answer length | 64.0 (hit budget) | 33.9 | 16.5 | tokens generated before stopping (or the budget, if it never stops). The DPO number is the one §3 explains mechanistically. |
| dict MC accuracy | 26.5% | 29.5% | 33.0% | 4-way multiple-choice by log-likelihood; chance = 25%. Still close to the phase-6 "near-floor for a 10M model" regime (see the eval-suite deep dive) — read the +3.5-pt DPO bump as directionally-nice, not a confident quality claim; the probe is small (200 examples) and this model never had much definitional knowledge to begin with (Part A's own finding). |
| pretrain val ppl | 34.93 | 40.10 (+14.8%) | 44.82 (**+28.3% vs base**) | plain CE on the frozen books+dictionary val set — the forgetting probe, unchanged mechanic since Part A/`SFTTrainer`. The DPO-only increment is `44.82/40.10 − 1 = +11.8%`, in just 91 steps — compounding fast, consistent with §2's drift-magnitude story. |

---

## 6. The three-part throughline: what SFT, LoRA, and DPO each actually changed

A synthesis worth keeping, since all three parts fine-tuned the *same* base toward the *same*
broad goal but changed genuinely different things:

- **SFT (Part A) taught a protocol.** Before it, the model had no notion of "a turn ends" — it
  just continued text. SFT's assistant-only loss mask taught *when to stop*, a first-order,
  binary-ish lesson (supervise these tokens, not those). The objective (cross-entropy against one
  known-good target per example) has no contrastive structure — nothing to exploit a shortcut
  against, because there's no "worse" example in the loss at all.
- **LoRA (Part B) taught the SAME lesson, cheaper and more reversibly.** Same protocol, same
  cross-entropy objective, just fewer trainable parameters (a frozen base + low-rank adapters).
  Quality was competitive-to-better; the real story was optimizer-memory cost (13-53× cheaper) and
  the frozen base's reversibility.
- **DPO (Part C) taught something qualitatively different: a *preference* between two already
  reasonable-looking candidates,** operating *within* the "answer and stop" behavior SFT already
  established (both `chosen` and `rejected` are chat-formatted, stop-token-terminated responses —
  DPO never re-taught the protocol). This is a genuinely harder learning problem: instead of
  "match this one target," it's "rank these two, and generalize the *reason* you ranked them that
  way to unseen pairs." **That contrastive structure is exactly what opens the door to shortcut
  learning** (§3) — SFT's plain cross-entropy has no "other" example to be fooled by; DPO's whole
  mechanism is built from comparing two examples, so whatever axis most cheaply separates them
  (here: length) gets learned fastest, correct lesson or not. This is the same principle behind
  "hard negative mining" in contrastive/metric learning more broadly (a concept from outside LLMs
  you likely already have intuition for): a contrastive objective is only as good as how well its
  negative examples isolate the *intended* axis of variation, and an "easy" negative (one that
  differs on some other, cheaper-to-detect axis too) teaches the wrong lesson faster than the right
  one.

---

## 7. Answers to the session's explicit ask (index)

- *Is there more to learn from Part C beyond notes.md/D-053?* → Yes, principally §3c: the
  aggregate 95.8%/8.97 numbers hide a real, uneven story across failure modes that changes the
  interpretation from "DPO worked, with a length caveat" to "DPO learned the easy (length) lesson
  much faster and harder than the hard (factual) lesson, and the aggregate margin is dominated by
  the easy one" — a materially sharper, more actionable finding.
- *What is DPO's loss, really?* → §1. Logistic regression / binary cross-entropy where the two
  "features" are the model's own before/after log-likelihood ratios for the two candidate
  responses, and the fixed "weight" is `β`.
- *What does a reward_margin of 8.97 actually mean?* → §2. `margin/β` nats of average log-ratio
  gap; `exp()` of that is the implied odds ratio (~10^39 here) — a quantified "this drifted a lot,"
  not a vibe.
- *Is the length confound a bug, or does it mean DPO learned nothing real?* → §3. Neither — it's a
  well-documented, literature-backed phenomenon (Singhal et al. 2023, SimPO, Rafailov et al. 2024),
  and the new per-mode breakdown shows real content learning (`wrong_fact` 87.2%) alongside the
  length-driven part, not instead of it.
- *Why did the run stop early, and is that a real reason or an excuse?* → §4. A real, checked
  (batch-shape ruled out), honestly-not-root-caused slowdown, combined with an already-saturated
  signal — a legitimate cost/benefit call, documented as such rather than hidden.

---

## 8. Takeaways (revision checklist)

1. **DPO = RLHF's KL-regularized objective, solved in closed form, substituted into Bradley-Terry,
   turned into a logistic-regression-shaped loss.** No reward model, no RL loop — `β` plays the
   role of a regularization strength / inverse temperature throughout.
2. **`reward_margin / β` is a log-ratio; exponentiate it to read the implied odds ratio** — a
   concrete way to see "how much has this drifted," not just a number going up.
3. **A reference-relative reward is length-*resistant*, not length-*proof*** — it cancels a
   length-scaled floor only to the extent policy and reference share it; genuinely new drift can
   still correlate with length.
4. **Break contrastive-training results down by the axis you constructed, not just the aggregate.**
   The 95.8%/8.97 headline numbers looked like a clean win; splitting by failure mode revealed the
   model was 12× more confident on the easy (length-correlated) distinction than the hard
   (fact-correlated) one — the real lesson of this session.
5. **Contrastive objectives are only as good as their negative examples isolate the intended
   axis.** 2/3 of our "bad" examples were also longer; the model partly learned "shorter,"
   because that was the cheapest generalization consistent with 2/3 of the supervision.
6. **This is documented DPO behavior, not an implementation bug** — SimPO's length-normalized
   reward and the direct-alignment overoptimization scaling-law paper both study this exact
   phenomenon, including the small-model-overoptimizes-faster pattern this run reproduced.
7. **Stopping early on a real, checked (not guessed) signal is a legitimate research decision** —
   ruling out the obvious alternative explanation (batch shape) before accepting "probably
   MPS/thermal" as the working hypothesis is what makes it a documented finding instead of an
   excuse.

## 9. Open questions / flagged follow-ups (→ parking lot, refines the existing D-053 item)

- **Length-normalize the reward for any future DPO run** (SimPO-style: divide `sequence_logprobs`
  by token count before computing `reward_chosen`/`reward_rejected`) — now a concrete,
  literature-backed fix, not just "cap verbose's length."
- **Rebalance or reweight the failure-mode mix** before a longer run: given `wrong_fact` is both
  the intended core lesson AND the one with the weakest gradient signal (§3c), either oversample
  it, or explicitly upweight its loss term, so the model doesn't finish learning length faster than
  facts.
- **Root-cause the MPS slowdown properly** (bisect: run policy-only forward passes at DPO's typical
  sequence length to isolate "long sequences" from "two live models" as the driver) before trusting
  wall-clock estimates on a bigger/longer run.
- **A complete-epoch re-run**, once the above land, to get a clean end-of-epoch number — and, given
  §3c, to check whether `wrong_fact`'s accuracy keeps climbing with more steps (real learning
  continuing) or plateaus while the length-driven modes' margins keep inflating (pure
  over-optimization from here on).

## 10. Links

- Decisions: D-053 (Part C); D-051/D-052 (Parts A/B, for the throughline in §6).
- Run: `experiments/20260719_p8_dpo-s-dictionary/` (`notes.md`, `metrics.jsonl`, `eval_dpo.json`,
  `samples/step_000050.txt` — the terse/vague qualitative examples referenced in §3c/§5).
- Report: `docs/results/finetune_report.md` Part C. Derivation: `notebooks/09_dpo_from_scratch.ipynb`.
- Papers: Rafailov et al. 2023, [DPO](https://arxiv.org/abs/2305.18290) (NeurIPS 2023); Bradley &
  Terry 1952 (pairwise-comparison ranking, the statistical model DPO's loss is built on); Singhal,
  Goyal, Xu, Durrett 2023, [*A Long Way to Go: Investigating Length Correlations in
  RLHF*](https://arxiv.org/abs/2310.03716); Meng, Xia, Chen 2024,
  [SimPO](https://arxiv.org/abs/2405.14734) (NeurIPS 2024); Rafailov, Chittepu, Park, Sikchi,
  Hejna, Knox, Finn, Niekum 2024, [*Scaling Laws for Reward Model Overoptimization in Direct
  Alignment Algorithms*](https://arxiv.org/abs/2406.02900) (NeurIPS 2024); Gao, Schulman, Hilton
  2022, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760)
  (ICML 2023) — the original (PPO-era) version of the same over-optimization phenomenon.
