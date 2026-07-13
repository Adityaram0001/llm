# 20260713_p5_s-wave-d-zloss — z-loss (PaLM '22)

**Hypothesis:** z-loss (`coeff * mean(logsumexp(logits)^2)`, coeff=1e-4) penalizes the softmax
normalizer growing unbounded — a stability aid for large-scale/long training, not expected to
change quality at this small scale where logits are already well-behaved.

- **Result:** val_loss 3.5029 (ppl 33.21), **+0.0052 vs control** — within the 0.015-0.02 (D-035) noise
  floor. No measurable effect, in either direction.
- **Conclusion:** NULL RESULT, as expected. At S-tier/98M tokens there's no logit-blowup failure
  mode for z-loss to fix, so it neither helps nor hurts — consistent with the technique's actual
  purpose (long-horizon/large-scale stability, e.g. PaLM's 540B-param, trillion-token regime).
  Revisit at longer training budgets or bf16-precision-sensitive setups where logits could
  plausibly drift.
