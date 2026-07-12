# 20260711_p4_s-lr-sweep-hi

**Hypothesis:** lr=3e-3 (3x the D-021 baseline default of 1e-3) vs the same schedule at
0.3x/1x -- part of phase 4's lr-sweep lesson, specifically the "watch overshooting happen on
purpose" half of it.

**Observation:** 300 steps, 19.66M tokens. val_loss trajectory (steps 0/50/.../250): 9.40, 6.23,
5.48, 5.14, 4.95, 4.84 -- consistently worse than lr-sweep-mid (1e-3) at every checkpoint, but
never diverged (no NaN/Inf at any point) and actually ended with the *lowest* mean grad_norm of
the 3 sweep runs (0.566 vs mid's 0.687 and lo's 1.055).

**Conclusion:** `grad_clip=1.0` did its job -- it bounded the *damage* from an lr that's too
large (no blowup), but it did not rescue the *outcome* (still clearly worse than lr=1e-3). The
lower grad_norm despite worse loss is the interesting bit: bigger updates per step likely pushed
the model into a flatter-gradient region that is nonetheless a worse loss value, rather than a
literal instability. Lesson for future sweeps: "didn't diverge" is not the same as "was a good
choice" -- always compare against the actual winner's curve, not just against NaN.
