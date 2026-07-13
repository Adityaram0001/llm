# Wave D deep dive: optimizers, schedules, and what 13 short runs actually taught us

*Discussion session, 2026-07-13, right after the Wave D implementation/run session (D-039). This
note exists so you don't have to re-read 13 notes.md files + the registry + DECISIONS.md D-039
to remember what happened and why — everything here traces back to those artifacts (linked at
the bottom) if you want the raw numbers again.*

## The shape of the wave, in one paragraph

Wave D asked five separate questions in one batch of cheap runs (~98.3M tokens each, ~2-7 min on
the RTX 5090): does the *optimizer* matter (AdamW vs Muon vs Lion)? Does the *lr schedule shape*
matter (cosine vs WSD vs constant)? Does *z-loss* help? Does *grad clipping* actually protect
anything? Does *batch size* trade off against optimizer steps the way theory says it should? The
honest answer, run by run: **one huge real effect (Muon), one very clean real effect (the
schedule hierarchy), a handful of expected nulls, and two results that needed an honest caveat
rather than a clean headline (Lion, the 1M-batch point).** That mix — not every ablation being a
clean win — is itself the lesson: real experiments produce confounds and caveats, and the
discipline is reporting them rather than reaching for the cleanest-sounding story.

## 1. Why Wave D got its own control instead of reusing `p4_s_baseline`

Every other Wave D delta is measured against `20260713_p5_s-wave-d-control` (val_loss 3.4977),
not directly against `p4_s_baseline` (3.5037) — even though they're the *same* hyperparameters
(lr=1e-3, betas=[0.9,0.95], wd=0.1, cosine, grad_clip=1.0, seed 1337). The only thing that
changed is `micro_batch`/`grad_accum`: 16/8 (the Mac-tuned default) became 64/2 (the RTX 5090's
measured sweet spot, D-030) — same 65,536 tokens/step effective batch either way.

Why does that matter enough to need a new control? `MixedSourceLoader` draws its random offsets
from `data_step = step * grad_accum + micro` (see `loader.py`) — it's what makes resume
bit-exact (D-023) without needing to save any sampler state. But it also means **the exact
tokens seen at training step N depend on `grad_accum`**, even when `micro_batch * grad_accum`
(the effective batch) is held constant. Step 10 of a `grad_accum=8` run and step 10 of a
`grad_accum=2` run are looking at different slices of the corpus. So the two recipes aren't
bit-comparable — they're two different (but both valid) samples from the same distribution,
which is exactly what the D-035 noise floor (0.015-0.02) exists to judge: is a 0.006 gap ("Wave D
control" vs `p4_s_baseline`) real or noise? It's noise. Good — the substitution is safe, and
every Wave D run is ~4x cheaper to run than it would have been under the Mac-tuned batch config
(same lesson Wave C's n_heads=4 control taught: when a "just for this GPU" tweak changes what
data a run sees, mint a fresh control rather than assuming it's still comparable to the old one).

## 2. Muon — the biggest single effect in the project so far

**Mechanism.** Muon (Keller Jordan, 2024, from the nanoGPT speedrun project) is momentum SGD
with one extra step: before the update is applied, it's passed through a **Newton-Schulz
iteration** that pushes the update matrix's singular values all towards 1 — i.e. it
*orthogonalizes* the update. Why would you want that? A raw gradient (or raw momentum) update
to a weight matrix isn't "isotropic" — some directions in weight-space get a much bigger nudge
than others, because gradient magnitude is dominated by whichever singular direction the loss
landscape currently curves most steeply in. Orthogonalizing means every direction moves by
*about the same amount* per step, so directions that would otherwise be starved of update
(small gradient component, but still meaningful for the loss) get to move too. This is
specifically for **2D weight matrices** (attention/FFN projections) — it doesn't make sense for
embeddings (a lookup table, not a matrix whose "directions" mean anything geometric) or norm
gains (a single scalar-per-channel scale, no matrix structure to orthogonalize). That's why the
recipe is a *hybrid*: Muon for the 2D hidden matrices, a small separate AdamW for everything else
(`_build_optimizers` in `trainer.py`, `muon_lr=0.02` vs the aux AdamW's `lr=1e-3`).

The orthogonalization itself (`zeropower_via_newtonschulz5`, `src/llmlab/train/optimizers.py`) is
a quintic iteration `X <- aX + bX(X^T X) + cX(X^T X)^2` with tuned constants `(a,b,c) =
(3.4445, -4.7750, 2.0315)` — it converges an arbitrary matrix's singular values towards 1 in ~5
iterations using only matmuls, no actual SVD (which would be far too slow to run every optimizer
step). It's an approximation, not exact orthogonalization — our own test
(`test_newtonschulz5_output_is_near_semi_orthogonal`) shows singular values landing roughly in
[0.68, 1.12] after 5 steps on a small 32x48 test matrix, not exactly at 1 — "tightly clustered,"
not "perfectly orthogonal." That's expected and fine; the point is *evening out*, not
perfection.

**What we measured.** val_loss 3.3432 vs control's 3.4977 — **delta -0.1545**, more than 10x the
noise floor. This is, numerically, the single largest effect size found anywhere in the project
to date (bigger than QK-norm's -0.062 in Wave A, bigger than ALiBi's -0.021 in Wave B). The
trajectory is the more interesting part though: the gap was **-0.267 at 500 steps, -0.185 at
1000 steps, -0.155 at the end** — Muon's advantage is *biggest early and shrinks over training*,
though it never closes. That pattern matches how the nanoGPT speedrun community actually talks
about Muon: it's framed as "gets you to a target loss in fewer steps," a convergence-*speed*
claim, not "raises the ceiling you'd eventually reach with enough steps." Our data is consistent
with exactly that framing — if we'd trained twice as long, a reasonable guess (not proven here)
is the gap would keep narrowing rather than staying at -0.155 forever.

**Caveat to remember:** `muon_lr=0.02` was Jordan's commonly-cited default, not something we
swept for this model/token-budget. The size of the win is real (way past the noise floor either
way), but the *exact* number could move with a proper muon_lr sweep — a natural Wave-D-followup
if you want to squeeze more out of it before committing it to the phase-9 recipe.

## 3. Lion — a result that needed an asterisk, not a headline

**Mechanism.** Lion (Chen et al. 2023, discovered via program search rather than hand-derived)
replaces AdamW's per-parameter adaptive step size with `sign(beta1 * momentum + (1-beta1) *
grad)` — every parameter moves by exactly `lr` each step, in the direction the sign says, no
more, no less. Only one momentum buffer is kept (vs AdamW's two: `exp_avg` and `exp_avg_sq`), so
optimizer state is half the memory. But because the update magnitude no longer reflects the
gradient's actual scale (a gradient of 0.0001 and a gradient of 100 produce the *same* step size,
just possibly different sign — verified directly in
`test_lion_update_magnitude_is_lr_regardless_of_gradient_scale`), the *lr itself* has to do more
work, which is why the paper recommends a much smaller lr (~3-10x smaller than the AdamW recipe
it replaces) paired with a much larger weight decay (~3-10x larger) to compensate.

**What we measured.** val_loss 3.9203 — **delta +0.4226**, the worst of the wave, using the
paper's suggested conversion (lr 1e-3 -> 3e-4, wd 0.1 -> 0.3) applied once, not swept. Looking at
the loss curve, Lion is behind from the very first logged checkpoint and the gap never narrows —
that shape (uniformly behind, not "diverges then recovers" or "converges then falls behind") is
more consistent with "the lr scale is simply off for this model" than "Lion doesn't work here."
**Read this as "our one-shot hyperparameter guess underperformed," not "Lion loses to AdamW/Muon"**
— a real verdict would need at least a small lr sweep (e.g. try 1e-4, 3e-4, 1e-3 and see where
the curve actually sits relative to control), which we didn't spend the session's time budget on.

## 4. The schedule hierarchy — the wave's cleanest, most surprising finding

**The three shapes** (`_schedule_multiplier` in `trainer.py`):
- **cosine** (control): warmup, then continuous cosine decay from step ~30 all the way to
  `max_steps`, ending at `lr * lr_min_ratio`.
- **constant**: warmup, then flat at peak lr *forever* — no decay at all.
- **WSD** (Warmup-Stable-Decay, Hu et al. 2024 / MiniCPM): warmup, then flat at peak lr for most
  of training ("stable"), then a short linear decay to `lr_min` only in the last
  `wsd_decay_ratio` fraction of steps (we used 0.2, i.e. decay only in the final 300 of 1500
  steps).

**What we measured, and why the *order* of events matters as much as the final numbers:**

| schedule | final val_loss | delta vs control |
|---|---|---|
| WSD | 3.3764 | **-0.1213** |
| constant | 3.4303 | **-0.0674** |
| cosine (control) | 3.4977 | — |

The hierarchy **WSD > constant > cosine** only makes sense once you look at *when* each schedule
is doing what. At step 500 and step 1000, WSD is already slightly ahead of cosine — and WSD's
explicit decay phase doesn't even start until step 1200! So WSD's early advantage has nothing to
do with decaying better; it's that **cosine has been decaying continuously since step ~30**,
while WSD (and constant) are still sitting at full peak lr. Full lr, for longer, means faster
raw progress per step — cosine is trading that away starting almost immediately, in exchange for
a smoother approach to a low final lr. The data says that trade is a net loss here: even
`constant`, which *never* decays at all, beats cosine's gradual decay. But `constant` still
loses to WSD, because WSD gets the best of both — hold the high-lr regime as long as possible
(like constant), *then* spend a short, sharp decay at the very end (which constant never does) to
settle into a lower-loss point. The intuitive summary: **decaying early is worse than not
decaying at all; decaying briefly at the very end is better than either.**

This is a clean, small-scale reproduction of exactly the argument WSD's original paper (MiniCPM)
makes for why it's replacing cosine as the default schedule in a lot of newer training recipes.

## 5. The WSD multi-budget bonus — training one checkpoint, deciding the budget later

This is WSD's other headline property, separate from "it trains better": **because the stable
phase never decays, you don't have to decide your total token budget before you start training.**
With cosine, the decay curve is baked in from step 1 — if you don't know `max_steps` in advance
(or want to change your mind later), you either guess wrong (and either waste an over-long decay
tail or get cut off before decay finishes) or you have to restart from scratch. With WSD, you can
just... keep training in the stable phase, and decide when to decay whenever you want.

We demonstrated this for real, not just as a thought experiment: `wave_d_constant`'s run (warmup
+ stable forever, no decay — it doubles as "WSD with the decay never triggered") finished its
1500 steps at val_loss 3.4303, still sitting at peak lr. We then took that *exact* checkpoint and
forked it two different ways via `scripts/train.py --resume`, each fork writing a fresh
`config.yaml` with `schedule=wsd` and `wsd_decay_ratio` set so the decay starts exactly at the
resume point (step 1500):

| fork | extra steps | extra tokens | final val_loss | delta vs the shared checkpoint |
|---|---|---|---|---|
| short | +150 | +9.83M (+10.0%) | 3.3220 | **-0.1083** |
| long | +400 | +26.2M (+26.7%) | 3.2768 | **-0.1535** |

Same starting weights both times, just a different decay-tail length — decided *after* the
stable-phase investment was already made, not before. The longer fork (3.2768) ends up the best
single number anywhere in Wave D, edging out even Muon (3.3432), though that's not a perfectly
fair comparison since it used more total tokens (124.5M vs the fixed ~98.3M budget everything
else in the wave used).

**One limitation worth remembering if you want to repeat this at a bigger scale:** `Trainer`
only keeps `latest.pt` (overwritten every `checkpoint_every` steps) and `best.pt` (overwritten on
improvement) — it does NOT keep numbered per-step snapshots. So we could only fork from the ONE
checkpoint that survived to the end of `wave_d_constant`'s run (step 1500), not from multiple
different points along its stable phase. A more thorough demo (forking from, say, step 750 *and*
step 1500 of the same run) would need the trainer to save numbered snapshots at intermediate
points, which it currently doesn't — worth a small trainer feature if a future phase wants to
lean on this harder (e.g. phase 9's capstone deciding its compute budget adaptively).

## 6. z-loss — a null result that's exactly what the mechanism predicts

**Mechanism.** z-loss (PaLM, 2022) adds `coeff * mean(logsumexp(logits, dim=-1) ** 2)` to the
loss. `logsumexp(logits)` is `log(Z)`, the log of the softmax normalizer — cross-entropy only
cares about *differences* between logits (softmax is shift-invariant), so nothing in the main
loss stops the whole logit vector from drifting arbitrarily large or small together. At large
enough scale (PaLM's 540B params, huge token counts, bf16/int8 quantization sensitivity), that
drift can become a real numerical stability problem. z-loss directly penalizes `log(Z)` growing
large, pinning it down without changing which token wins.

**What we measured.** val_loss 3.5029 vs control's 3.4977 — delta +0.0052, inside the noise
floor. **This is the expected result, not a disappointing one**: at 10M params and 98M tokens,
nothing in this project has ever shown logit blowup or numerical instability (every wave so far
trains cleanly). z-loss has nothing to fix here, so it neither helps nor hurts — consistent with
its actual purpose being a much-larger-scale stability aid. Worth revisiting specifically if a
future run shows genuine logit-scale drift (e.g. a much longer training run, or extreme learning
rates), not as a matter of routine.

## 7. Grad-clip-off — the spike that didn't happen, and why that's still informative

The phase-5 spec predicted this ablation would "watch it spike." It didn't, and the reason why is
itself a good lesson in reading training metrics carefully.

**First subtlety: the logged `grad_norm` metric can't show a difference by construction.**
`torch.nn.utils.clip_grad_norm_` always returns the gradient norm **before** any clipping is
applied — clipping only rescales the gradients in-place if that norm exceeds the threshold, it
doesn't change what gets returned. So `grad_norm` in `metrics.jsonl` is identical whether
`grad_clip=1.0` or `grad_clip=1e6` (effectively off) — both runs show the exact same peak (5.51
at step 0, see `wave_d_optimizers_schedules.png` panel d). If you only looked at that one metric
you'd conclude clipping "does nothing," which is wrong — it's just the wrong metric to look at
for that question. The real effect of clipping shows up in what happens to the *loss* after a
large gradient, since that's the thing clipping actually changes (the scaled-down update that
gets applied).

**Second: even looking at the right thing (loss), the effect was real but small, not a spike.**
`gradclip_off`'s train_loss was consistently ~0.02-0.1 *higher* than control's at every early
checkpoint (step 10 through 70) — a steady, persistent gap, not a single catastrophic event. Final
val_loss delta: +0.0215, just past the noise floor. **Why no dramatic blowup?** This model is
15 layers, pre-norm, with a 30-step lr warmup — pre-norm residual streams are specifically known
to be much more forgiving of large early gradients than post-norm (Wave A's D-036 already showed
post-norm's failure mode is *stagnation*, not blow-up, at this same depth), and the warmup means
the lr is tiny exactly when the one large early gradient (step 0's norm-5.5 spike) occurs — a
big gradient times a tiny lr just isn't that dangerous. **The lesson:** grad clipping's value
here is "cheap, consistent, small insurance," not "the thing standing between you and NaN losses"
— a dramatically different-looking result (an actual spike) would likely need a less forgiving
setup: no warmup, a much higher lr, or a much longer run giving a rare large gradient more chances
to land at a moment when it actually matters.

## 8. The batch-size study — confirming the coupling, with an honest confound

**Mechanism (the "linear scaling rule" backdrop).** At a FIXED total token budget, a bigger
effective batch means fewer optimizer steps (`total_tokens / (batch_size * seq_len)`). If you
don't scale the learning rate up to compensate, each of those fewer steps is still only as big a
move as before — so you simply take fewer, equally-sized steps through the same total data,
which undertrains relative to a smaller-batch run that got many more (smaller) updates. This
wave deliberately did NOT apply the scaling rule, specifically to make that undertraining
visible.

**What we measured** (fixed ~98.3M token budget throughout):

| effective batch | steps | final val_loss | delta vs control |
|---|---|---|---|
| 65,536 tok/step (control) | 1500 | 3.4977 | — |
| 262,144 tok/step | 375 | 4.2567 | +0.759 |
| 1,048,576 tok/step | 94 | 5.3942 | +1.8965 |

Monotonic, as predicted. **But the 1M-batch point has a real confound worth remembering:**
`warmup_steps=30` was left the same absolute value for every Wave D config, including this one —
so 30 of this run's 94 total steps (32%!) were spent ramping the lr up from near-zero, versus
control's 30-of-1500 (2%). Some of the 1M point's badness is genuinely "16x fewer optimizer
steps at a fixed lr" (the intended variable), but part of it is simply "almost a third of this
run's whole budget was spent below peak lr," which is a config-generation oversight, not a
property of large batches. The 0.25M point (30/375 = 8% warmup, much more proportionate) is the
cleaner data point for this specific lesson. **Practical takeaway for any future batch-size
study:** scale `warmup_steps` proportionally to `max_steps` (e.g. ~5-10% of total steps) rather
than holding it fixed in absolute step count when steps-per-run varies a lot across the sweep.

## 9. AdamW hyperparameter sweep (weight_decay, beta2) — expected nulls

- **wd=0 vs control's wd=0.1:** delta -0.0042, within noise. Weight decay's regularizing effect
  (shrinking rarely-useful weights towards zero to reduce overfitting) needs actual overfitting
  pressure to show up — at only 98M tokens (roughly one pass over the small S-tier corpus, not
  many repeated epochs), there isn't much overfitting happening yet for wd to correct.
- **beta2=0.999 vs control's beta2=0.95:** delta +0.0122, within noise (borderline-low end,
  closest of the wave's nulls to the floor). `beta2` sets the effective averaging horizon for
  AdamW's second-moment (gradient-variance) estimate — `0.95` averages over roughly the last ~20
  steps, `0.999` over roughly the last ~1000. In a short, 1500-step run, a horizon of ~1000 steps
  is a meaningful fraction of the *entire* run, so in principle it should adapt more sluggishly
  early on than `0.95`'s fast-adapting estimate — but that theoretical disadvantage doesn't clear
  the noise floor here. Keep `beta2=0.95` (D-021's original choice) un-overridden; this isn't
  evidence it's wrong, just evidence this particular comparison isn't powered to detect a
  difference at this budget.

## 10. What actually changes in the phase-9 recipe

- **Muon is now the single strongest lever this project has found**, if training wall-clock/
  compute is the thing you're optimizing for — worth a proper `muon_lr` sweep before the
  capstone, given how much headroom the un-tuned default already found.
- **Prefer decaying the lr only late (WSD), or not at all (constant), over cosine's continuous
  early decay** — this is a free win with zero extra implementation cost once the schedule
  dispatch exists (it already does, `OptimConfig.schedule`).
- **Lion needs a real sweep before it's usable as a recommendation either way** — right now it's
  an open question, not a settled "worse than AdamW" verdict.
- **z-loss and grad-clip are "keep them, can't prove their value yet at this scale"** — cheap
  insurance whose payoff (if any) would show up at longer runs / bigger models / more extreme
  hyperparameters, not here.
- **The batch/steps/lr coupling is a config-authoring lesson, not just a research finding**: any
  future sweep spanning a wide range of `max_steps` needs `warmup_steps` scaled proportionally,
  not held at a fixed absolute value.

## 11. Does any of this break re-running Waves A/B/C, or scaling up to bigger models?

Short version: **no risk for fresh runs or bigger models, one real (but currently unexercised)
break for resuming old checkpoints.** Longer version, since this came up as a direct question
this session:

- **`OptimConfig`'s new fields** (`optimizer`, `schedule`, `z_loss_weight`, `muon_*`,
  `wsd_decay_ratio`) all ship with defaults that reproduce the exact pre-Wave-D behavior
  (`optimizer="adamw"`, `schedule="cosine"`). Old Wave A/B/C YAML configs don't set these fields
  at all, so loading them today (`TrainConfig.from_yaml`) fills in those defaults and behaves
  identically to how they ran originally — verified by the refactored cosine-schedule math being
  algebraically identical to the pre-refactor formula (see the `lr_at_step`/
  `test_lr_at_step_linear_warmup_then_cosine_decay` test, unchanged and still passing). **Fresh
  reruns of any old wave, or of a bigger M/L-tier config written from scratch, are unaffected.**
- **The checkpoint file format did change**: `save_checkpoint` used to write one
  `"optimizer_state_dict"` key; it now writes a list under `"optimizer_state_dicts"` (needed so
  Muon's two optimizers — the Muon instance and its auxiliary AdamW — both get saved/restored).
  `load_checkpoint` would `KeyError` trying to resume a checkpoint saved by the OLD trainer code.
- **This isn't currently exposed**: every Wave A/B/C run finished without needing a resume, and
  none of their checkpoints were ever pulled back to the Mac (per CLAUDE.md's "don't hoard
  checkpoints" rule, they stayed on now-stopped remote instances). So there's no old-format `.pt`
  file anywhere that a resume attempt would actually hit.
- **Recommendation (not yet acted on):** don't build a backward-compatible shim into
  `load_checkpoint` preemptively — that's speculative complexity for a scenario that doesn't
  currently exist. If it's ever actually needed (some old checkpoint resurfaces and needs
  resuming), the fix is a five-line one-off script: load the old `.pt` with `torch.load`, wrap its
  `optimizer_state_dict` value in a list under the new key, `torch.save` it back out. A minute of
  work, only when the need is real.

## Links

- Decision log: `docs/DECISIONS.md` D-039 (full design rationale + impacts list)
- Per-wave summary: `docs/results/ablation_log.md`, "Wave D" section
- Figure: `docs/results/wave_d_optimizers_schedules.png` (`scripts/plot_wave_d.py`)
- Registry rows: `experiments/registry.csv`, all `20260713_p5_s-wave-d-*` run_ids
- Per-run hypothesis/observation/conclusion: `experiments/20260713_p5_s-wave-d-*/notes.md`
- New code: `src/llmlab/train/optimizers.py` (`Lion`, `Muon`, `zeropower_via_newtonschulz5`),
  `src/llmlab/train/trainer.py` (`_build_optimizers`, `_schedule_multiplier`,
  `_split_params_by_ndim`), `src/llmlab/train/config.py` (`OptimConfig`'s new fields)
- Papers: Jordan '24 (Muon / nanoGPT speedrun), Chen et al. '23 (Lion), Hu et al. '24 / MiniCPM
  (WSD), Chowdhery et al. '22 / PaLM (z-loss)
