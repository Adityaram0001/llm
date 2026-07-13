# 20260713_p5_s-wave-d-batch_1m — batch-size study, 1M eff. tokens/step

**Hypothesis:** same as batch_025m but more extreme — 16x control's effective batch, only 94
total steps for the same ~98.3M token budget.

- **Result:** val_loss **5.3942** (ppl 220.12), **+1.8965 vs control** — real, dramatically worse.
- **Confound worth flagging honestly:** `warmup_steps=30` was NOT scaled down for this run —
  30/94 = 32% of this run's entire budget is spent in warmup (vs control's 30/1500 = 2%). Some of
  this run's severe undertraining is a genuine "16x fewer optimizer steps at a fixed lr" effect
  (the intended variable), but part of it is an artifact of an oversized warmup fraction eating
  into an already-tiny step budget — a design oversight in this session's config generation, not
  a property of large batches per se.
- **Conclusion:** REAL DIRECTIONALLY (bigger batch without lr scaling undertrains at fixed
  tokens — same lesson as batch_025m, more extreme), but the MAGNITUDE here is inflated by the
  warmup confound above. Treat batch_025m as the cleaner data point for this study; if this
  comparison matters again, scale `warmup_steps` proportionally to `max_steps` (e.g. ~5-10% of
  steps) rather than holding it fixed in absolute step count.
