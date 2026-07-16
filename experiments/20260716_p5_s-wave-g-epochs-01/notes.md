# 20260716_p5_s-wave-g-epochs-01 — Wave G multi-epoch overfitting lab, 1 epoch

**Role:** Books-only pool (no dictionary — `data/clean/books`, 14,141,233 tokens, D-045), so
short structured dictionary entries don't confound a pure "prose memorization" study. Trained
for exactly 1 epoch over that pool (216 steps @ 65,536 tok/step ≈ 14.16M tokens). Eval against
the matching `books_only_val.bin` (2 held-out books), not the general val set, for an
apples-to-apples train/val comparison on the SAME distribution the model trained on.

- **Result:** train_loss=5.427, val_loss=5.695 (ppl 297) — gap=+0.268. Both still high;
  1 epoch is nowhere near enough to fit even this small pool.
- **Conclusion:** the undertrained reference point for the epochs=4/16 comparison — see D-045.
