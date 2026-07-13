# 20260713_p5_s-wave-d-wsd_branch_long — WSD multi-budget bonus, long decay fork

**Role:** the SAME fork point as wsd_branch_short (`wave_d_constant`'s step-1500 checkpoint,
3.4303) but a longer decay tail — 400 more steps (+26.2M tokens, +26.7%) — demonstrating that a
bigger post-hoc budget decision, from the exact same shared stable investment, buys further
improvement.

- **Hypothesis:** a longer decay tail from the same checkpoint should beat the short fork.
- **Result:** val_loss **3.2768** (ppl 26.49) — **-0.1535 vs the constant checkpoint**, and the
  BEST single number in all of Wave D (better even than Muon's 3.3432, though at +26.2M more
  tokens than the fixed-98.3M-budget comparisons, so not a fully apples-to-apples ranking).
- **Conclusion:** REAL. Together with wsd_branch_short, this is a clean small-scale reproduction
  of WSD's core practical pitch (Hu et al. '24, MiniCPM): train ONE stable-phase checkpoint, then
  decide the final token budget after the fact by choosing how long to decay — cheaper than
  committing to a schedule shape/length up front and re-training from scratch for each budget.
