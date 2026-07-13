# 20260713_p5_s-wave-d-wsd — WSD schedule (Hu et al. '24, MiniCPM)

**Hypothesis:** warmup -> stable (flat lr) -> linear decay over the last 20% of steps should
match or beat cosine at equal budget, since WSD's stable phase doesn't waste steps on early decay.

- **Result:** val_loss **3.3764** (ppl 29.26), **-0.1213 vs control's cosine** — real, well past
  the 0.015-0.02 (D-035) noise floor, second-best result of the wave after Muon.
- **Trajectory (see wave_d_optimizers_schedules.png panel b):** WSD is already slightly *ahead*
  of cosine by step 500-1000 (still in its stable, undecayed phase -- decay only starts at step
  1200) — cosine's continuous decay from step 30 onward appears to cost it real ground even
  BEFORE WSD's explicit decay tail kicks in and pulls further away in the last 300 steps.
- **Conclusion:** REAL, ROBUST WIN. Confirms WSD's central claim (front-loaded cosine decay wastes
  the high-lr/fast-progress phase). See the WSD multi-budget bonus (wsd_branch_short/long) for
  the schedule's other headline property.
