# 20260713_p5_s-wave-d-adamw_wd0 — AdamW weight_decay=0.0

**Hypothesis:** removing weight decay (vs control's 0.1) shouldn't matter much at only 98M
tokens/one pass over a small corpus — decay's regularization value shows up more with longer
training / more overfitting pressure.

- **Result:** val_loss 3.4935 (ppl 32.90), **-0.0042 vs control** — within the 0.015-0.02 (D-035) noise
  floor.
- **Conclusion:** NULL RESULT, as expected. wd=0.1 (D-021's default) is not disprovable OR
  provable at this token budget either way — this run doesn't argue for changing it, just
  confirms it isn't costing anything at this scale. Revisit at a longer training budget (M/L
  tier, more epochs) where overfitting pressure would actually engage weight decay's mechanism.
