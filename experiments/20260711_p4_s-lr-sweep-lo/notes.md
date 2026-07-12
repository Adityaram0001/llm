# 20260711_p4_s-lr-sweep-lo

**Hypothesis:** lr=3e-4 (0.3x the D-021 baseline default of 1e-3) vs the same schedule at 1x/3x
-- part of phase 4's lr-sweep lesson (watch under/over-shooting happen on purpose).

**Observation:** 300 steps, 19.66M tokens. val_loss trajectory (steps 0/50/.../250): 9.59, 6.99,
5.95, 5.52, 5.34, 5.25 -- monotonically improving but consistently the worst of the 3 sweep
runs at every single checkpoint, not just at the end. grad_norm mean 1.055 (highest of the 3),
33% of steps had raw grad_norm > 1.0 (i.e. clipping engaged).

**Conclusion:** not unstable -- undertrained. A lower lr takes smaller steps per update, so at
this short a budget (300 steps) it simply hasn't covered as much loss-landscape ground as
lr=1e-3. No evidence this lr is "bad" in an absolute sense, only that it's slower to converge
at equal step count. See `20260711_p4_s-lr-sweep-mid`'s notes for the winner and D-025 for the
sweep's overall conclusion.
