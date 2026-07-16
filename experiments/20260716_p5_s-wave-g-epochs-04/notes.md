# 20260716_p5_s-wave-g-epochs-04 — Wave G multi-epoch overfitting lab, 4 epochs

**Role:** Same books-only pool/val split as the 1-epoch run, 4x the budget (864 steps ≈ 56.6M
tokens ≈ 4 epochs). See `20260716_p5_s-wave-g-epochs-01`'s notes for the full setup rationale.

- **Result:** train_loss=4.104, val_loss=4.448 (ppl 85.5) — gap=+0.344 (up from +0.268 at 1
  epoch). Val loss has already dropped close to where it ends up plateauing by 16 epochs
  (4.13-4.45 range) — most of the achievable val improvement on this small pool happens in the
  first few epochs.
- **Conclusion:** the gap is opening, consistent with the phase-5 spec's prediction — see the
  16-epoch run and D-045 for the full curve.
