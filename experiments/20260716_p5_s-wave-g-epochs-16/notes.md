# 20260716_p5_s-wave-g-epochs-16 — Wave G multi-epoch overfitting lab, 16 epochs

**Role:** Same books-only pool/val split, 16x the 1-epoch budget (3456 steps ≈ 226.5M tokens ≈
16 epochs). See `20260716_p5_s-wave-g-epochs-01`'s notes for the full setup rationale.

- **Result:** train_loss=3.207 (still falling at the end), val_loss=4.128 (ppl 60.8, actually
  BETTER than the 4-epoch run's 4.448) — gap=+0.921, by far the widest of the three.
- **Conclusion:** textbook overfitting signature — train loss keeps dropping (the model keeps
  memorizing the repeated 14.14M-token pool) while val loss has essentially plateaued/stopped
  improving between 4 and 16 epochs (4.448 → 4.128, a much smaller improvement than 1→4 epochs'
  5.695 → 4.448). The train/val GAP is what "opens" here, exactly as the phase-5 spec predicted
  — val loss itself never gets worse at this scale/budget (contrast with the scaling-law runs,
  D-045, where the larger models' val loss does turn around and rise). See D-045 for the full
  3-point curve and its relationship to the scaling-law finding.
