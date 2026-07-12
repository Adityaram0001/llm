# 20260711_p4_s-lr-sweep-mid

**Hypothesis:** lr=1e-3 (the D-021 baseline default, tested here as the sweep's control) vs
0.3x/3x -- part of phase 4's lr-sweep lesson (watch under/over-shooting happen on purpose).

**Observation:** 300 steps, 19.66M tokens. val_loss trajectory (steps 0/50/.../250): 9.48, 5.88,
5.23, 4.96, 4.81, 4.73 -- **strictly ahead of both lr-sweep-lo (3e-4) and lr-sweep-hi (3e-3) at
every single checkpoint**, not just at the final step. grad_norm mean 0.687, only 10% of steps
had raw grad_norm > 1.0 (least clipping-engaged of the 3).

**Conclusion:** **Winner.** 1e-3 is the right lr at this scale, and the margin over both
neighbors is consistent across the whole run, not a fluke at one checkpoint. This ratifies
D-021's original default (chosen from the nanoGPT-tiny-scale analogy + GPT-3's lr-vs-params
formula) rather than overriding it -- see D-025. Used as-is for `20260711_p4_s-baseline`.
