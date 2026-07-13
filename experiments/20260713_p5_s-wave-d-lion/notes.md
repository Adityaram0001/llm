# 20260713_p5_s-wave-d-lion — Lion optimizer (sign-based update)

**Hypothesis:** Lion (Chen '23) with the paper's recommended hyperparameter conversion from
control's AdamW recipe (lr 1e-3 -> 3e-4, i.e. /3.3; wd 0.1 -> 0.3, i.e. x3) should land near
control, trading a bit of quality for half the optimizer-state memory.

- **Result:** val_loss **3.9203** (ppl 50.41), **+0.4226 vs control** — real, and clearly the
  worst optimizer of the wave, well beyond the 0.015-0.02 (D-035) noise floor.
- **Reading:** this is a ONE-SHOT hyperparameter guess (a single paper-recommended lr/wd
  conversion, not a swept optimum) — the gap is most plausibly Lion being meaningfully
  under/over-tuned for this specific model/token-budget rather than a fundamental "Lion is worse"
  finding. The loss curve (wave_d_optimizers_schedules.png, panel a) shows Lion consistently
  behind from the very first checkpoint and never catching up, which looks more like "wrong lr
  scale" than "wrong optimizer" — a well-tuned Lion run would likely close most of this gap.
- **Conclusion:** REAL result AS RUN, but not a fair verdict on Lion itself — flag as
  "needs an lr sweep before drawing a real conclusion" rather than "Lion loses to AdamW/Muon" in
  the phase-9 recipe. Do not read this as ruling Lion out.
