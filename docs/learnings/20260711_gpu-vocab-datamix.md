# GPU choice, batch tuning, vocab scaling, and data mixing — four coupled decisions

*Discussion session 2026-07-11 (pre-phase-4), with the planning model. Questions: 5090 vs
RTX PRO 6000; GPU-utilization monitoring & auto batch sizing; 16k→32k vocab + bigger model;
finance/self-help domain share in the corpus.*

## 1. RTX 5090 vs RTX PRO 6000 — VRAM you can't use is money burned

Both are Blackwell (sm_120): same CUDA generation, same Docker image, same code — **zero
code/build changes to switch between them** (that's what D-010's device-agnostic rules bought
us; per-hardware differences like micro-batch are already YAML keys, not code).

The sizing math: our L-tier (105M params) in fp32 = 0.42GB weights + 0.42GB grads + 0.84GB
AdamW states ≈ 1.7GB; even with huge activation batches we'd struggle to *fill* the 5090's
32GB. The PRO 6000's 96GB buys nothing for a 105M model at ~2× the hourly rate — VRAM is the
constraint for 7B+ models, not ours. **Rule: rent for $/FLOP, not for VRAM, until the model
or batch actually needs the memory.** (Niche exception: 96GB could host several concurrent
ablation processes on one card, but scheduling complexity isn't worth it for us.)

## 2. GPU utilization: monitor tokens/sec, calibrate batch ONCE, never auto-adjust mid-run

- **Ground truth is tokens/sec** (already in our metrics.jsonl every log step), not
  `nvidia-smi`'s util% — util can read 100% while memory-bandwidth-bound. wandb online mode
  also auto-charts GPU util/power/VRAM for free.
- **The feedback loop is a pre-run calibration, not a runtime controller** (D-018):
  `scripts/find_batch_size.py` sweeps micro-batch (doubling until OOM or plateau) for ~2 min
  on the actual hardware, reports the tokens/sec sweet spot, you set the config, launch.
  Phase 0 proved why measuring beats assuming: MPS had a 3–15× throughput cliff at ~1GB
  alloc, far below any advertised limit.
- **Why NOT dynamic batch adjustment during training:** effective batch size is a
  *hyperparameter* — it changes gradient noise and interacts with LR. A run whose batch
  drifts mid-flight is scientifically uninterpretable and incomparable to its baseline;
  our whole phase-5 lab depends on one-variable ablations. The portable pattern instead:
  **effective batch is fixed in the config; micro-batch × grad-accum factorizes it per
  hardware** (Mac: 8×32, 5090: 64×4 — same optimization trajectory, different packaging).

## 3. 32k vocab + 160–180M params + 1.6× data — right math, wrong time (parked for v2)

The 1.6× data scaling instinct is exactly right (Chinchilla ~20 tok/param: 160M → 3.2B).
But: (a) changing vocab retokenizes everything and **breaks perplexity comparability with
every run so far** (only bits-per-byte survives a tokenizer change); (b) phase 2 measured
only **49.3% of a 32k vocab even fires** on our current corpus — 32k earns its embedding cost
only after the corpus grows/diversifies (FineWeb-Edu + finance books will help exactly that).
So: finish v1 at 16k/105M, and "v2: 32k vocab, 160–180M, ~3.2B tokens" is now in PROGRESS.md's
parking lot — it's a coherent scale-up *after* the phase-5 recipe is known. (Note: at 160M
with d_model 768, a tied 32k embedding is ~24.6M ≈ 15% — healthy.)

## 4. Domain data mix (finance + wisdom) — the % that matters is of the TRAINING STREAM

The core insight this discussion surfaced: **corpus % ≠ training %.** What steers the model
is the share of *effective tokens seen during training*, controlled by the loader's per-source
mixing weights (phase 4), not by how many books sit on disk. Current raw reality: books+dict
= 17.7M tokens ≈ <1% of a 2.1B-token stream if mixed naturally — invisible. Steering requires
deliberate upsampling.

But upsampling has a ceiling: Muennighoff '23 — repeated data holds value up to ~4 epochs,
decays after. So "finance+wisdom at 20% of stream" = 420M effective tokens needs ≥~100M raw
domain tokens to stay under 4 epochs. Getting there: (a) more PD books (Gutenberg-era
finance/wisdom classics — e.g. Franklin's *Way to Wealth*, Clason's *Richest Man in Babylon*
(1926), Smiles' *Self-Help* (1859), Allen's *As a Man Thinketh*, Wattles, Adam Smith —
**modern self-help/finance bestsellers are copyrighted, off-limits**); (b) filter a
finance/econ-heavy slice OUT of FineWeb-Edu (keyword/classifier pass — it contains plenty);
(c) data-factory synthetic finance prose/Q&A.

Recommended split of the ~2.1B stream (starting point, to be ablated):
**~70–75% general supplement** (fluency comes from here — a small model that skips broad
English is incoherent, defeating the domain goal), **~15–25% domain** (wisdom/philosophy +
finance/self-help, ≤4 epochs), **~3–5% dictionary**. And the honest caveat: pretraining mix
biases *knowledge*; **SFT (phases 7–8) is where conversational steering mostly happens** —
finance-flavored instruction data will do more for "can chat about money wisdom" than +10%
pretraining share. Both levers, in that order.

This is also now a first-class experiment: P5-G gains a **domain-mix ablation** (e.g. 10% vs
25% vs 50% stream share → measure domain probes vs general val loss — the classic
specialization-vs-generality tradeoff), and phase 6 gains **finance/wisdom probes** alongside
the dictionary probes. Tracked as RW-4 (corpus expansion + mixing weights; final % is the
user's call at phase-4 time).

## Related
D-010 (device-agnostic), D-015 (tiers/data budget), D-017 (cloud logistics), D-018 (GPU choice
& static-batch policy), RW-4, parking lot (v2 scale-up) · Papers: Chinchilla '22,
Muennighoff '23 (repeat ceiling), Gururangan '20 (domain-adaptive pretraining), TinyStories/
phi line (narrow+clean beats broad+noisy at small scale).
