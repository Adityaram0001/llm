# 20260712_p5_s-wave-a-gelu — Wave A run 3/4: SwiGLU -> GELU MLP (param-matched)

**Hypothesis:** SwiGLU's gating mechanism (Shazeer '20) should give a modest loss improvement
over a plain GELU MLP at *matched parameter count* (ffn_mult 8/3 for SwiGLU's 3 matrices vs 4.0
for GELU's 2, both landing at 8*d_model^2 FFN params — see `configs/model_s_gelu.yaml`'s comment
for the derivation) — matching most published ablations.

**Observation:** GELU finished at val_loss 3.6764 (ppl 39.50) vs SwiGLU baseline's 3.5037 (ppl
33.24) — delta +0.1727, and the gap is consistent (~0.15-0.2) at every checkpoint from step 200
onward, not narrowing or noisy. Actual param counts: GELU model 9,727,872 vs SwiGLU baseline
9,713,472 (0.15% larger, well within the intended match).

**Conclusion:** a **real, robust** effect, an order of magnitude past the 0.015 noise floor
(D-035). SwiGLU's gated linear unit gives a clear quality edge over plain GELU at equal params,
confirming the literature and validating D-016's choice of SwiGLU as the S/M/L-tier default.
