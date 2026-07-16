# 20260716_p5_s-wave-f-moe-biasfree — DeepSeekMoE, DeepSeek-V3 aux-loss-FREE (bias) balancing

**Hypothesis:** Same architecture as `20260716_p5_s-wave-f-moe-auxloss` (8 routed + 1 shared
experts, top-2, active-param-matched to the dense control), but load balance comes from a
per-expert bias added to routing SELECTION logits only (never to the combine weight, never part
of any loss — DeepSeek-V3 S2.1.2), updated by a small fixed-size sign-based nudge once per
optimizer step. Prediction: comparable final quality to the aux_loss variant, since both are just
different ways of encouraging the same balanced-routing outcome; the real difference should show
up in balancing DYNAMICS (bias-free has no gradient pressure, so should react more slowly to
imbalance) rather than final loss.

- **Result:** val_loss 3.4149 (ppl 30.41), **-0.0828 vs control (3.4977)** — clearly REAL (>4x
  noise floor), same DeepSeekMoE capacity win as the aux_loss variant.
- **aux_loss vs bias_free, head to head:** 3.4149 vs 3.4070 = **0.0079 apart — within the
  0.015-0.02 noise floor, not distinguishable.** At S-tier / this token budget, the choice of
  balancing method does NOT measurably affect final quality.
- **Where the two methods DO differ — balancing speed (this is the real, measured finding):**

  | step | aux_loss std/mean | bias_free std/mean |
  |------|-------------------|---------------------|
  | 10   | 0.236             | 0.406               |
  | 200  | 0.026             | 0.166               |
  | 400  | 0.022             | 0.102               |
  | 600  | 0.012             | 0.075               |
  | 800  | 0.019             | 0.023               |
  | 1000 | 0.009             | 0.015               |
  | 1490 | 0.011             | 0.009               |

  aux_loss reaches good balance (std/mean < 0.03) by step ~200; bias_free doesn't get there
  until step ~800-1000, and is briefly the MORE unbalanced of the two mid-training (visible as
  the persistent bright band for one expert in `docs/results/wave_f_deepseek_specials.png`'s
  bottom-right heatmap). This is exactly the mechanistic trade-off DeepSeek-V3's method predicts:
  the aux loss is a gradient that reshapes routing every step in proportion to how imbalanced the
  current batch is; the bias update is a small FIXED-size nudge (`bias_update_rate=0.001`)
  regardless of imbalance magnitude, so it takes many more steps to correct a large initial
  imbalance — a slower, bounded control loop, in exchange for zero gradient interference with
  the main objective (V3's actual motivation for the method: no aux loss competing with LM loss).
  By the end of training BOTH reach comparably tight balance (~0.01 std/mean) and comparable
  quality — bias_free "catches up" given enough steps.
- **Conclusion:** aux-loss-free balancing reproduces its DeepSeek-V3 selling point at S-tier —
  equal final quality to the gradient-based aux loss, with the tradeoff visible as a slower
  balancing ramp rather than a quality cost. See D-044 for a real evaluation-metric bug this wave
  caught and fixed before either verdict was written (both numbers above are post-fix).
