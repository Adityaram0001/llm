# 20260713_p5_s-wave-e-mb32_accum4 — micro-batch/accum equivalence, mb=32 accum=4

**Hypothesis:** effective batch size (micro_batch × grad_accum × seq_len) is the hyperparameter
that matters for the loss trajectory — D-018's "effective batch is fixed, factorization
re-derives per hardware" rule predicts this run should match the control's loss (same effective
batch, 128 seqs/step) even though it's split into more, smaller micro-steps.

- **Quality:** val_loss 3.4985 (ppl 33.07), **+0.0008 vs control (3.4977)** — indistinguishable,
  well within noise. Confirms the hypothesis directly.
- **Speed:** ~248,222 tok/s — the **slowest run in the whole wave**, ~45% slower than control's
  ~455,133 tok/s, despite doing the exact same total FLOPs. 2x the micro-steps of the control (at
  half the size) means 2x the Python-loop/kernel-launch/data-sampling overhead per optimizer
  step — consistent with D-022's finding that this model's throughput is launch-overhead-bound
  at S-tier, not compute-bound.
- **Conclusion:** loss is invariant to batch/accum factorization (as it should be, mathematically
  equivalent gradient accumulation), but wall-clock is NOT — smaller micro-batches are strictly
  worse here with no compensating benefit. See `mb128_accum1` for the opposite end of this axis.
