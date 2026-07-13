# Wave E deep dive: what "efficiency" actually costs and buys, with the numbers behind each claim

*Discussion session, 2026-07-13, right after the Wave E implementation/run session (D-040). This
note goes past what's in the 6 notes.md files, `ablation_log.md`, and D-040 — it re-derives a few
things from the raw metrics/CSVs that weren't spelled out at run time (why memory grows linearly
not quadratically with seq_len, where the 5090's kernel-launch overhead actually shows up, why
torch.compile's first log line reads 3,581 tok/s, and why a param-matched weight-tying rerun is
harder than "just run it again"). Everything traces back to the artifacts linked at the bottom.*

## The shape of the wave, in one paragraph

Wave E asked five efficiency questions, and **four of the five were designed to be NULL results
on loss** — bf16, gradient checkpointing, batch/accum factorization, and torch.compile are all
supposed to be mathematically transparent (same forward/backward math, different execution
strategy). Getting a null on loss for those isn't a disappointing result, it's the correctness
check passing. The actual findings live in **speed and memory numbers** instead, and unlike
Waves A-D (config + run + analysis, using machinery already built in phase 3/4), this wave needed
genuinely new trainer/model code — `precision`, `gradient_checkpointing`, and `compile` didn't
exist before this session. The one axis that WASN'T designed to be a null (weight tying) produced
the wave's messiest result — real, but confounded by an unmatched parameter count, which turns
out to be hard to fix cleanly (more on that below).

## 1. bf16 vs fp32 — why the loss doesn't move but the speed does

**What `autocast_ctx` actually changes.** `torch.autocast(device_type=..., dtype=torch.bfloat16)`
does NOT make the model "a bf16 model." The parameters stay fp32 the entire time (check
`model.parameters()` dtype under either setting — unchanged). What autocast does is wrap specific
ops (matmuls, convolutions) so that *their inputs* get cast to bf16 for the compute, while
numerically fragile ops (softmax, norm reductions, loss) stay in fp32. Gradients that flow back
through a bf16-computed matmul still accumulate into fp32 `.grad` tensors. So bf16 autocast is
best understood as "run the expensive matmuls at reduced precision, keep everything else and all
persistent state at full precision" — not a wholesale precision downgrade.

**Why that predicts the result we got.** If the *only* thing that changes is intermediate matmul
precision, and bf16 has enough dynamic range for this model's activation magnitudes (bf16 has the
same exponent range as fp32, just fewer mantissa bits — it can't lose catastrophically to
underflow/overflow the way fp16 sometimes does), there's no reason to expect a quality difference
at all. That's exactly what we saw: val_loss 3.5060 (fp32) vs 3.4977 (control, bf16) — a
**+0.0083 delta, half the noise floor's lower bound (0.015)**. Not just "not significant" but
comfortably inside noise.

**Why the speed difference is real and substantial.** The RTX 5090's tensor cores have dramatically
higher throughput for bf16 matmuls than fp32 ones (this is a hardware fact, not a software
choice — it's the same reason mixed-precision training became standard practice once GPUs shipped
tensor cores). Measured: ~296.8K tok/s (fp32) vs ~455.1K tok/s (control, bf16) — **fp32 is ~35%
slower**, paying for zero measurable quality benefit. This is D-009's "bf16-by-default" choice,
now verified rather than assumed on the actual training model instead of taken on faith from
general mixed-precision literature.

## 2. Gradient checkpointing — the ~1.72x number, and a pattern that wasn't in the original writeup

**Mechanism recap.** Normal backprop keeps every intermediate activation tensor alive in memory
from the forward pass until backward has used it (last layer's activations get used first,
working backward — so everything has to sit in memory simultaneously in the worst case).
Checkpointing (Chen et al. '16) trades that away: instead of keeping a block's internal
activations, it only keeps the block's *input*, and when backward reaches that block, it
**re-runs the forward pass for just that block** to regenerate what it needs, then discards it
again immediately after computing that block's gradient. More compute (one extra forward pass per
checkpointed segment), less peak memory (only ever holding one block's worth of internals at a
time, not all fifteen).

**The clean part: it's exact, not approximate.** val_loss 3.4889 (checkpointed) vs 3.4977
(control) — delta -0.0088, within noise, confirming what the mechanism predicts: recomputing the
exact same forward pass produces the exact same numbers (mod floating-point non-associativity from
a different op-execution order, which `test_gradient_checkpointing_matches_no_checkpointing`
confirms is negligible — atol=1e-6 on every gradient).

**The memory curve — and a pattern worth flagging that the original notes.md didn't dig into.**

| seq_len | no checkpointing | with checkpointing | ratio |
|---------|------------------:|--------------------:|------:|
| 128  | 2,856.2 MB  | 1,664.6 MB  | 1.716 |
| 256  | 5,598.1 MB  | 3,269.8 MB  | 1.712 |
| 512  | 11,124.0 MB | 6,480.1 MB  | 1.717 |
| 1024 | 22,175.9 MB | 12,900.7 MB | 1.719 |
| 2048 | OOM         | 25,742.0 MB | —     |

The ~1.72x ratio being *constant* across seq_len is itself informative, but look at how memory
scales with seq_len **within** each column: 128→256→512→1024 roughly **doubles memory at every
doubling of seq_len** (ratios 1.960, 1.987, 1.994 for the uncheckpointed line; 1.964, 1.982,
1.991, 1.995 checkpointed). That's **linear growth in seq_len, not quadratic.**

This is worth pausing on, because "attention is O(seq_len²) in memory" is the standard mental
model (the attention score matrix is literally `seq_len × seq_len` per head) — if that quadratic
term dominated, doubling seq_len should roughly **quadruple** memory once seq_len is large enough
to matter, not double it. It doesn't. The most likely explanation, grounded in the actual code
(`attention.py` calls `F.scaled_dot_product_attention` with no explicit backend pinning — no
`torch.nn.attention.sdpa_kernel` context, no `torch.backends.cuda.enable_flash_sdp` calls): PyTorch's
SDPA dispatcher is almost certainly selecting a **flash-attention-style kernel** on the 5090 given
bf16 + causal masking, which computes attention **without ever materializing the full seq_len×seq_len
score matrix** — it fuses the QKᵀ, softmax, and ×V steps into a kernel that keeps memory linear in
seq_len (Dao et al. '22's whole point). We didn't set this deliberately; it's the SDPA dispatcher's
default choice on this hardware, and this benchmark is the first time this project has actually
looked closely enough at memory-vs-seq_len to notice it.

**A second, smaller pattern**: the ratio climbs slightly toward 2.0 as seq_len grows (1.960 →
1.994), rather than sitting exactly at 2.0 throughout. That's consistent with `peak_memory ≈
fixed_overhead + linear_term × seq_len` — a small seq_len-independent floor (model params,
optimizer state, CUDA context) that matters proportionally more at short seq_len and washes out
as the linear activation term grows. Fitting that two-parameter model to the 128/256 points
predicts the 512/1024 points to within ~0.5-1%, which is a reasonable fit for back-of-envelope
purposes (not a rigorous profile — a real memory profiler would separate these terms exactly, but
we don't need more precision than this to draw the right conclusion).

**Why the OOM boundary matters more than the ratio.** The ~1.72x reduction is a nice number, but
the practically decisive fact is: **uncheckpointed OOMs at seq_len=2048 (mb=64), checkpointed
reaches 2048 fine and OOMs at 4096.** Checkpointing doesn't just shrink memory by a constant
factor — because memory scales with seq_len, a constant-factor reduction directly translates into
"one more doubling of usable context length" before hitting the 5090's 32GB ceiling. That's the
actual lever a future long-context run (the phase-9 capstone's chat-context goal, RW-5 part b)
would be pulling.

**Napkin math: where does that many GB actually come from?** The S-tier model is 9.71M params.
At fp32 (master weights): 9.71M × 4 bytes ≈ 38.8MB. Gradients, same shape, same dtype: another
~38.8MB. AdamW's two per-parameter moment buffers (`exp_avg`, `exp_avg_sq`), both fp32: 2 ×
38.8MB ≈ 77.7MB. **Total params + gradients + optimizer state ≈ 155MB.** Compare that to the
*smallest* peak-memory measurement in the whole sweep: 2,856MB at seq_len=128. Even there,
activations account for **(2856 - 155) / 2856 ≈ 94.6%** of peak memory — everything that isn't
activations is a rounding error. This is exactly why gradient checkpointing (which targets
activations specifically) is such a disproportionately effective lever compared to, say, a
lower-memory optimizer (Lion, Wave D, halves the optimizer-state term) — at this model size, the
optimizer-state term was never the problem to begin with.

## 3. Micro-batch/grad-accum equivalence — loss is free, wall-clock isn't, and a counter-intuitive twist

**Why the math guarantees loss equivalence.** Gradient accumulation computes the loss/gradient on
each micro-batch separately, divides by `grad_accum` (see `train_step`'s `loss = loss /
self.cfg.batch.grad_accum`), and sums the results before the optimizer step — this is
*algebraically identical* to computing the gradient on one big batch of `micro_batch × grad_accum`
sequences at once, AS LONG AS there's no batch-size-dependent statistic involved (like BatchNorm's
running mean/var, which this model doesn't have — it's RMSNorm throughout, which normalizes
per-example, not across the batch). So there was never any *reason* to expect a loss difference
between mb=64/accum=2, mb=32/accum=4, and mb=128/accum=1 — this ablation is really a test of
whether the implementation actually does what the math says it should, and it does (deltas
+0.0008 and +0.0040, both deep inside noise).

**Why the wall-clock difference is real and large.** ~248.2K tok/s (mb=32/accum=4, slowest) vs
~525.1K tok/s (mb=128/accum=1, fastest) — **more than 2.1x apart for identical total FLOPs.**
Every micro-step pays a roughly fixed cost regardless of its size: Python-loop overhead in
`train_step`'s `for micro in range(...)` loop, a CUDA kernel-launch queue for every op in the
forward/backward graph, and `MixedSourceLoader.get_batch`'s CPU-side memmap indexing. mb=32/accum=4
does **4x as many micro-steps** as mb=128/accum=1 for the same total work, so it pays that fixed
cost 4x as often. This is D-022's phase-4 finding (measured on the Mac: throughput flat across
micro_batch 1-32, "kernel-launch overhead and unified-memory traffic dominate over raw compute" at
this model size) now reproduced and *quantified* on CUDA hardware instead.

**The counter-intuitive part worth internalizing:** a faster GPU makes this overhead *relatively*
worse, not better. The 5090 does real compute for a given micro-step very quickly — so as
micro-step size shrinks, the fixed per-step overhead (which doesn't get any smaller) becomes a
*larger fraction* of an ever-shrinking per-step compute time. On a slower GPU (or a bigger model,
where each micro-step's real compute takes longer), the same fixed launch overhead would be a
smaller relative tax. This is exactly why D-022's Mac numbers already showed this pattern at
micro_batch as large as 8-16 (MPS unified memory + a less powerful GPU has different but
comparably-sized fixed overheads relative to its own compute speed) — the *lesson* generalizes
(fixed overhead always matters more, relatively, on faster hardware / smaller models) even though
the *specific numbers* (which micro_batch is "large enough") don't transfer directly between Mac
and 5090.

**The takeaway that\'s now backed by a real number, not just a rule of thumb:** always factorize
toward the *largest* micro-batch that fits in memory, minimizing `grad_accum` rather than treating
it as a free dial — at this model size on this GPU, the difference between getting that right and
getting it wrong is over 2x wall-clock for zero quality difference either way.

## 4. torch.compile — a real 18% win, and the one number that shows exactly where it comes from

**Mechanism, briefly.** `torch.compile` traces the model's forward pass into a computation graph
(via TorchDynamo), then hands that graph to a backend compiler (TorchInductor by default) which
fuses multiple small ops into fewer, larger CUDA kernels and can eliminate redundant Python-level
dispatch overhead entirely for the traced region. The first time the compiled function actually
runs, that trace-and-compile step happens — a one-time cost paid up front, amortized over every
subsequent call with the same input shapes.

**We can actually see that one-time cost in the raw log, not just infer it.** Every run logs
`tokens_per_sec` every 10 steps. The *first* logged value for every OTHER Wave E run is
~93K-106K tok/s (a normal "still warming up the CUDA allocator" number). The compile run's first
logged value: **3,581 tok/s** — roughly 30x slower than every other run's first measurement. That's
not noise; it's the graph-capture/compilation step itself getting timed as part of that first
10-step interval. Steady-state after that point: mean 740,514 tok/s (min 659K, max 1.1M across the
remaining log points) — dramatically faster than every other run once compiled, which is exactly
what "big one-time cost, then faster forever after" should look like when you look at the
instantaneous per-interval numbers instead of the run-averaged wall-clock figure.

**Why the registered number (~535K tok/s, ~18% over control) is smaller than the steady-state
740K figure.** The registry/notes.md numbers are `tokens_trained / wall_hours` — the WHOLE run's
wall-clock, including the one-time compile cost, eval passes (every 100 steps), sample generation
(every 200 steps), and checkpoint writes (every 250 steps), none of which benefit from
`torch.compile` (compilation only wraps the model's `forward`, not `generate`, not the eval loop's
uncompiled-but-still-called-through-the-compiled-model forward, not checkpoint I/O). The
steady-state 740K figure is the pure train-loop compute rate; the ~535K figure is what you'd
actually experience end-to-end for a real run of this length. Both are "correct," they're just
measuring different things — worth remembering when comparing any two throughput numbers in this
project: check whether it's a wall-clock average (includes everything) or a steady-state
in-loop rate (compute only).

**A correctness detail worth understanding, not just the speed number.** `torch.compile(model)`
returns a wrapper object (`OptimizedModule`), and depending on the PyTorch version, calling
`.state_dict()` on that wrapper can produce keys prefixed with `_orig_mod.` instead of the
original parameter names — meaning a checkpoint saved from a compiled model might not load cleanly
into an uncompiled one (or vice versa) without extra key-remapping. Rather than depending on
whatever the currently-installed PyTorch version happens to do, `Trainer.__init__` now keeps
`self._raw_model` pointing at the pre-compile module permanently, and `save_checkpoint`/
`load_checkpoint`/`num_params()` all go through that reference instead of `self.model` (which
becomes the compiled wrapper only when `compile: true`). This means every checkpoint this project
writes has the same key format regardless of whether that particular run used `compile: true` —
checkpoints stay interchangeable across compiled and uncompiled runs, forever, without needing to
know or care what PyTorch's current wrapper behavior is.

## 5. Weight tying off — a real win, an honest caveat, and why the "obvious" fix is harder than it sounds

**Recap of the mechanism (D-016).** Tied embeddings (Press & Wolf '16) use the *same* matrix as
both the input token-embedding lookup and the final unembedding projection (the matrix that turns
the last hidden state into vocab-sized logits). The argument for tying: both directions are
answering a version of the same question ("which vector represents token X"), so sharing the
matrix halves the embedding parameter cost with no expected quality cost — and D-016 flagged this
mattering MORE at this project's 16k-vocab/small-model scale than at GPT-2/GPT-3 scale, since the
embedding table is a much bigger fraction of a 9.71M-param model (31.6%) than of a 1B+-param one.

**What we measured.** val_loss 3.4699 (untied, 12.79M params) vs 3.4977 (control, tied, 9.71M
params) — **delta -0.0278**, just past the 0.015-0.02 noise floor. Untied is better here, which on
its face reads as evidence *against* D-016's choice.

**Why that reading is premature — the params aren't matched.** The untied model has 31.6% more
total parameters (an extra, unshared 3.07M-parameter unembedding matrix) than the control. Giving
a model more capacity and watching it do better is not a surprising or new finding on its own —
the interesting question D-016 actually needs answered is "does tying cost quality **at a fixed
layer shape**," and this run doesn't isolate that variable.

**Why fixing this isn't a trivial "just run it again with different YAML."** To param-match, you'd
need to remove ~3.07M params from *somewhere else* in the untied model to bring it back to ~9.71M
total. Two obvious knobs, both awkward at this scale:
- **Cut layers.** Active (non-embedding) params per layer ≈ 442.9K (6.64M / 15 layers). Removing
  3.07M worth means cutting ~6.9 layers — roughly halving depth (15→8), which is not a small
  tweak, it's a different model shape entirely, confounding "does tying matter" with "does depth
  matter."
- **Shrink `ffn_mult`.** FFN params/layer = `3 × d_model² × ffn_mult` = `3 × 192² × 2.667 ≈
  294.9K` currently. Removing 3.07M total (204.8K/layer across 15 layers) means dropping
  `ffn_mult` from 2.667 to about **0.81** — less than a 1x-d_model hidden width, a severe and
  independently-quality-affecting FFN cut (Wave A already showed FFN width/activation choice is a
  real, non-trivial axis on its own).
- **Shrink `d_model`.** Constrained to multiples of 64 (`head_dim` is fixed, D-016), so the only
  smaller option is d_model=128 (2 heads instead of 3) — a 33% narrower model, again a different
  shape, not a small compensating tweak.

**The real lesson here, beyond the specific tying question:** this is the SAME D-016 finding
(embeddings are a disproportionately large fraction of a small model's budget) showing up twice —
once as the original argument for tying, and now again as the reason a clean param-matched
untied comparison is awkward to construct at S-tier. At a bigger model size (where the embedding
table is a much smaller % of the total, per D-016's own math), this same +3.07M would be a much
smaller relative perturbation and might be worth just absorbing into a "roughly matched" comparison
— worth remembering if this question resurfaces at M/L-tier rather than assuming the S-tier
difficulty generalizes.

## 6. Stacking the free wins — what phase 9 can actually expect

bf16 (already the default), torch.compile, and "largest micro-batch that fits" are all
**independently** free (zero measured quality cost) speed wins. They weren't measured *together*
this wave (each was tested in isolation against the same control), so the following is a
reasonable-assumption estimate, not a measured number:

- bf16 vs fp32: control is ~1.53x fp32's speed (455.1K / 296.8K)
- torch.compile vs uncompiled: ~1.18x (535.4K / 455.1K, wall-clock-inclusive figure)
- mb=128/accum=1 vs mb=64/accum=2: ~1.15x (525.1K / 455.1K)

If compile's and the batch-factorization's speedups are roughly independent multipliers on top of
the (already-bf16) control, stacking both gives **~1.18 × 1.15 ≈ 1.36x** over today's default S-tier
recipe — worth a real joint-measurement run before relying on this for an M/L-tier or phase-9
time/cost estimate, but a reasonable planning number in the meantime. **Gradient checkpointing is
deliberately NOT part of this stack** — it has a real, opposite-direction cost (~27% slower at
this size) and should only be reached for when memory, not speed, is the binding constraint.

## 7. Does any of this break re-running old waves, or change what "the recipe" means?

- **All three new `TrainConfig` fields ship with defaults that reproduce pre-Wave-E behavior**
  (`precision="bf16"`, `gradient_checkpointing=False`, `compile=False`) — every Wave A-D config
  loads and behaves identically today, verified by re-running the full local+remote test suites
  (89 local, 64 remote) after the change, all green.
- **The checkpoint-routing fix (`_raw_model`) is backward AND forward compatible** for anything
  that matters in practice: for any run with `compile=False` (every run except
  `wave_e_compile` itself), `self._raw_model is self.model` — identical object, so
  `save_checkpoint`/`load_checkpoint` behave exactly as before the fix. It only changes behavior
  for `compile=True` runs, which didn't exist before this session.
- **One thing this wave genuinely fixed, unrelated to training code:** a trailing-slash rsync bug
  during this session's remote sync (`rsync ... src/ configs/ ... dest/` copies `src/`'s
  *contents* into `dest/` rather than creating `dest/src/`) briefly left a stray, incomplete
  `llmlab/` package at the remote pod's repo root (missing the `data` subpackage entirely), which
  shadowed the real `src/llmlab/` via Python's cwd-first `sys.path` resolution and broke
  `tests/test_trainer.py`'s collection with a `ModuleNotFoundError`. Removed and re-synced
  correctly (no trailing slashes on the source args) — purely a remote-filesystem cleanup, no
  project code was ever wrong.

## Links

- Decision log: `docs/DECISIONS.md` D-040 (full design rationale + impacts list)
- Per-wave summary: `docs/results/ablation_log.md`, "Wave E" section
- Figure: `docs/results/wave_e_efficiency_memory.png` (`scripts/plot_wave_e.py`)
- Memory sweep raw data: `docs/results/wave_e_activation_memory.csv`,
  `docs/results/wave_e_activation_memory_gradckpt.csv` (`scripts/bench_activation_memory.py`)
- Registry rows: `experiments/registry.csv`, all `20260713_p5_s-wave-e-*` run_ids
- Per-run hypothesis/observation/conclusion: `experiments/20260713_p5_s-wave-e-*/notes.md`
- New code: `src/llmlab/train/config.py` (`precision`/`gradient_checkpointing`/`compile` fields),
  `src/llmlab/train/trainer.py` (`_autocast`, `_raw_model`, `compile_status`),
  `src/llmlab/model/gpt.py` (`gradient_checkpointing` attribute + block-wrap)
- Related earlier findings this wave reproduces/extends: D-009 (bf16-by-default), D-022 (D-022's
  Mac launch-overhead finding, now quantified on CUDA), D-016 (embedding parameter-budget math,
  now also explaining why param-matching the tying ablation is awkward)
- Papers: Chen et al. '16 (gradient checkpointing), Dao et al. '22 (FlashAttention — the likely
  explanation for the linear-not-quadratic memory curve), Press & Wolf '16 (weight tying)
