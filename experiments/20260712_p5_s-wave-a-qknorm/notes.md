# 20260712_p5_s-wave-a-qknorm — Wave A run 4/4: +QK-norm

**Hypothesis:** QK-norm (normalizing q/k per-head before the attention-score computation,
Gemma2/Qwen2-style) is usually framed as a stability aid that matters most at larger scale/depth
— going in, expected this S-tier/15-layer test to show at most a small or negligible effect.

**Observation:** QK-norm finished at val_loss 3.4414 (ppl 31.23) vs baseline's 3.5037 (ppl
33.24) — delta -0.0622, consistently 0.03-0.064 better at every checkpoint from step 100 onward,
with the gap *widening* over training (-0.038 at step 100 -> -0.062 at step 1400), not
shrinking. Extra params from the added per-head norm weights are negligible (9,715,392 vs
9,713,472, +0.02%).

**Conclusion:** a **real, robust, and the best result of Wave A** — well beyond the 0.015 noise
floor (D-035), and growing rather than fading with more training, which argues it's a genuine
optimization-quality effect (likely more stable attention-logit scale letting the rest of the
network train more effectively) rather than a lucky early boost. Worth carrying QK-norm forward
as a strong candidate default for later waves/tiers, contrary to the going-in expectation that
it would matter mainly at larger scale.
