# 20260713_p5_s-wave-d-wsd_branch_short — WSD multi-budget bonus, short decay fork

**Role:** demonstrates WSD's headline practical property — the "stable" phase doesn't need to
know the eventual total budget. This run RESUMES from `wave_d_constant`'s real step-1500
checkpoint (val_loss 3.4303, still at peak lr since that run never decays) and decays linearly to
lr_min over 150 more steps (+9.83M tokens, +10.0%), with `wsd_decay_ratio` set so decay starts
exactly at the resume point (step 1500).

- **Hypothesis:** a short decay tail off the shared stable checkpoint should recover most of
  WSD's decay benefit cheaply.
- **Result:** val_loss **3.3220** (ppl 27.72) — **-0.1083 vs the constant checkpoint it forked
  from**, for only +10% more tokens. Comparable to spending a full WSD run's decay tail (main
  `wave_d_wsd` run: -0.1213 vs cosine control) but achieved as a cheap ADD-ON after the fact,
  not planned in from the start.
- **Conclusion:** REAL, confirms the bonus's premise. One stable-phase training investment
  (`wave_d_constant`) produced a usable "decide your budget later" checkpoint that a short decay
  turns into a strong final model.
