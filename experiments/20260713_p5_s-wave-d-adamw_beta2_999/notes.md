# 20260713_p5_s-wave-d-adamw_beta2_999 — AdamW betas=[0.9, 0.999]

**Hypothesis:** beta2=0.999 (vs control's 0.95) averages the second-moment estimate over a much
longer horizon (~1000 steps vs ~20) — expected to matter more over long training; may be
slightly worse here since the longer horizon adapts slower to this run's relatively short
1500-step budget.

- **Result:** val_loss 3.5099 (ppl 33.45), **+0.0122 vs control** — within the 0.015-0.02 (D-035) noise
  floor (borderline low end).
- **Conclusion:** NULL RESULT. beta2=0.95 (D-021's default, tuned for a nanoGPT-tiny-scale short
  run) and beta2=0.999 (the GPT-3/more-conventional-scale default) are indistinguishable at this
  budget — the theoretical "slower-adapting beta2 costs you on a short run" story doesn't show up
  clearly enough to beat the noise floor. Keep beta2=0.95 as the default per D-021, un-overridden.
