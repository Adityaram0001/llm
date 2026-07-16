# 20260716_p5_s-wave-f-moe-auxloss — DeepSeekMoE, Switch-style aux-loss balancing

**Hypothesis:** DeepSeekMoE (DeepSeek-MoE '24) replaces the dense FFN with 8 fine-grained routed
experts + 1 always-on shared expert (top-2 routing), each expert sized so ACTIVE params/token
(shared + top_k = 3 expert-equivalents) match the dense control's single FFN. Total params grow a
lot (18.61M vs control's 9.71M) but active compute stays ~matched. Prediction: real quality win
from the extra total capacity, at ~equal active FLOPs/token to the control.

- **Result:** val_loss 3.4070 (ppl 30.17), **-0.0907 vs control (3.4977)** — clearly REAL, >4x
  the 0.015-0.02 noise floor (D-035). Confirms the DeepSeekMoE story at this tiny scale: more
  total capacity via many small experts, quality improves at matched active params/token.
- **Important measurement note (see D-044):** the FIRST attempt at this run reported val_loss
  3.5642 — apparently WORSE than the control — because `Trainer.evaluate()` was, at the time,
  reading `forward()`'s combined training loss (main CE + `aux_loss_weight * moe_aux_loss`)
  instead of pure CE. `moe_aux_loss` is summed across all 15 MoE layers (~1.0/layer at good
  balance, so ~15 in aggregate) and `aux_loss_weight=0.01`, adding ~+0.15 to the reported metric
  — almost exactly the apparent "gap" vs `bias_free` (which legitimately has zero aux loss by
  design, so its buggy number happened to already be correct by coincidence). Caught before any
  verdict was written; fixed in `GPT.forward`/`Trainer.evaluate` (now reads `last_aux_metrics
  ["ce_loss"]`, pure next-token CE, same convention as every other wave); re-ran clean. The
  buggy first attempt's run folder was deleted (never had notes.md/a verdict) rather than kept
  as a confusing duplicate.
- **Balancing dynamics (see the aux_loss-vs-bias_free pair's shared note in
  `20260716_p5_s-wave-f-moe-biasfree/notes.md`):** load balances FAST — std/mean across experts
  drops from 0.051 (init) to 0.026 by step 200, 0.012 by step 600, ~0.01 by the end. Gradient-
  driven balancing (the aux loss directly reshapes router logits every step) reacts quickly to
  imbalance.
- **Conclusion:** DeepSeekMoE's headline claim reproduces at S-tier — real quality win from
  fine-grained expert capacity at matched active params. Caveat carried from D-016/Wave E's
  weight-tying finding: this is a TOTAL-vs-active-params story, consistent with what the
  technique is actually supposed to buy (more knowledge storage, not more FLOPs/token), not a
  confound to apologize for.
