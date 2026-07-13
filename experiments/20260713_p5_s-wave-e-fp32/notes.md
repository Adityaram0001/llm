# 20260713_p5_s-wave-e-fp32 — precision: bf16 autocast off, plain fp32

**Hypothesis:** disabling bf16 autocast (full fp32 compute) should not change the loss
trajectory meaningfully but should be slower, since it forfeits the RTX 5090's tensor-core
throughput advantage for reduced-precision matmuls.

- **Quality:** val_loss 3.5060 (ppl 33.31), **+0.0083 vs control (3.4977)** — within the
  0.015-0.02 noise floor (D-035). NULL RESULT: bf16 and fp32 land in the same place.
- **Speed:** ~296,804 tok/s (98.3M tokens / 0.092h) vs control's ~455,133 tok/s (98.3M / 0.06h)
  — **~35% slower**. This is the real, expected cost of giving up tensor-core-accelerated
  matmuls.
- **Conclusion:** bf16 autocast is free accuracy-wise and a real, substantial speed win at this
  scale — confirms D-009/CLAUDE.md's bf16-by-default choice rather than just assuming it. No
  reason to ever train in fp32 on this hardware at this model size.
