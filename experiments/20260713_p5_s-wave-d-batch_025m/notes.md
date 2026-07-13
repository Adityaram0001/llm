# 20260713_p5_s-wave-d-batch_025m — batch-size study, 0.25M eff. tokens/step

**Hypothesis:** at a FIXED ~98.3M token budget, a 4x bigger effective batch (262,144 vs
control's 65,536 tok/step) means 4x fewer optimizer steps (375 vs 1500) — with lr deliberately
NOT rescaled, this should undertrain relative to control.

- **Result:** val_loss **4.2567** (ppl 70.57), **+0.759 vs control** — real, much worse.
- **Conclusion:** REAL, AS PREDICTED. Confirms that "more tokens per step" isn't free at a fixed
  token budget unless lr is scaled up to compensate for the fewer gradient updates (the standard
  "linear scaling rule" territory, deliberately not applied here as the point of this run). A
  clean, if unsurprising, confirmation of the batch/steps/lr coupling.
