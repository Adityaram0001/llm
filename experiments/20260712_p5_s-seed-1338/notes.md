# 20260712_p5_s-seed-1338 — seed-noise study, run 2/3

**Hypothesis:** identical config to `20260711_p4_s-baseline` (D-021 hyperparameters) except
`seed=1338` (vs baseline's 1337) should produce a val_loss within a small band of the baseline's
final 3.5037 — if not, the training pipeline is more seed-sensitive than expected and every later
Wave A-G ablation verdict would need a much wider "is this a real difference" threshold.

**Observation:** final val_loss 3.4970 (ppl 33.02) vs baseline's 3.5037 (ppl 33.24) — a 0.0067
difference, and the two curves track within ~0.007 at every logged eval checkpoint (100-step
intervals), not just at the end. Ran on the RTX 5090 (gpuhub) at ~126K tok/s vs the baseline's
~11.4K tok/s on Mac MPS — same config, same tokens/step, ~53x faster wall-clock (12.7 min vs
2.4hr) — first real (non-sweep) training confirmation that D-022's Mac throughput number and the
5090's sweep-only throughput numbers (D-032/D-034) both hold up in an actual training loop.

**Conclusion:** training is reproducible within a tight band across seeds at this scale/token
budget. See `20260712_p5_s-seed-1339`'s notes.md for the full 3-seed noise-floor computation
(mean/std/spread) that this run feeds into.
