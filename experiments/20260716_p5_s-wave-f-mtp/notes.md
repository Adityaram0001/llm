# 20260716_p5_s-wave-f-mtp — Multi-Token Prediction (+1 head predicting t+2)

**Hypothesis:** DeepSeek-V3 S2.2 adds sequential MTP heads that predict further-ahead tokens
using teacher-forced true embeddings, sharing the main output head — denser per-token training
signal, claimed to improve sample efficiency. This run adds exactly one extra head (predicts
t+2 from trunk hidden + true t+1 embedding), loss = main CE + 0.3 * mtp CE. The extra head is a
full transformer Block (+0.52M params, 9.71M -> 10.23M, +5.3%) but is TRAIN-TIME ONLY — dropped
at eval/generation, so inference compute is unaffected (mirrors DeepSeek-V3's own inference-time
behavior).

- **Result:** val_loss (main CE only, not mtp-weighted — see D-044) 3.5144 (ppl 33.6),
  **+0.0167 vs control (3.4977)** — right at the edge of the 0.015-0.02 noise floor (D-035).
  **Not clearly distinguishable from noise**, though if anything trending slightly worse, not
  better.
- **The MTP head's own loss (predicting t+2) is real and learns**: starts at 9.71 (~ln(16000),
  sane random-init baseline) and falls to 3.78 by the end of training — clearly not degenerate,
  the extra head is learning a real (if harder) task. It's notably HARDER than the main task
  (final mtp_loss 3.78 vs main CE 3.51) — expected, since predicting 2 tokens ahead from only
  one extra (teacher-forced) token of lookahead is intrinsically harder than 1-token prediction.
- **Conclusion:** at this scale/token budget (98.3M tokens, single extra head, S-tier ~10M
  params), MTP does NOT show a measurable benefit to the main next-token objective — the denser
  training signal doesn't clearly help, and the extra train-time compute (one more block's
  forward+backward per step) doesn't clearly hurt either, within noise. This does not contradict
  DeepSeek-V3's own result (measured at vastly larger scale/token budget, where sample-efficiency
  gains compound over much longer training) — it's consistent with a technique whose benefit
  needs either more scale, more tokens, or a larger `n_predict_tokens`/`loss_weight` sweep to
  surface at this lab's tier. Flagged as a candidate for a follow-up sweep (loss_weight,
  n_predict_tokens>1) if MTP is revisited, not concluded as a negative result on the technique
  itself.
