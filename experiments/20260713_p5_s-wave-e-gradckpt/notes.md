# 20260713_p5_s-wave-e-gradckpt — gradient checkpointing on

**Hypothesis:** gradient checkpointing (Chen et al. '16) recomputes activations on the backward
pass instead of storing them, so it should reproduce the exact same loss trajectory (it's a
recompute, not an approximation) while costing some wall-clock time. Its memory payoff won't
show up in THIS run since it uses the same seq_len=512/micro_batch=64 as the control — the real
test is `bench_activation_memory.py`'s seq_len sweep (see below).

- **Quality:** val_loss 3.4889 (ppl 32.75), **-0.0088 vs control (3.4977)** — within the noise
  floor. NULL RESULT as predicted: checkpointing doesn't change what's computed.
- **Speed (at this size):** ~333,000 tok/s vs control's ~455,133 — **~27% slower**, the real
  recompute cost, paid here for no memory benefit since 512/64 already fits comfortably.
- **Memory payoff (separate benchmark, `docs/results/wave_e_activation_memory{,_gradckpt}.csv`,
  fixed micro_batch=64, sweeping seq_len):**

  | seq_len | no checkpointing | with checkpointing | ratio |
  |---------|-----------------|---------------------|-------|
  | 128     | 2,856 MB        | 1,665 MB            | 1.72x |
  | 256     | 5,598 MB        | 3,270 MB            | 1.71x |
  | 512     | 11,124 MB       | 6,480 MB            | 1.72x |
  | 1024    | 22,176 MB       | 12,901 MB           | 1.72x |
  | 2048    | **OOM**         | 25,742 MB           | —     |
  | 4096    | —               | **OOM**             | —     |

  A remarkably consistent ~1.72x peak-memory reduction at every seq_len, and it buys exactly one
  more doubling of context length before hitting the 5090's 32GB ceiling (2048 fits checkpointed,
  OOMs uncheckpointed).
- **Conclusion:** textbook result. Checkpointing is quality-neutral, costs ~27% wall-clock at
  this size, and its real value is unlocking longer sequences / bigger batches under a fixed
  memory budget — exactly the trade-off Chen et al. describe, cleanly reproduced at S-tier scale.
  Worth reaching for whenever seq_len or micro_batch is memory-bound, not by default otherwise.
