# 20260712_p5_s-wave-a-layernorm — Wave A run 1/4: RMSNorm -> LayerNorm

**Hypothesis:** LayerNorm and RMSNorm should reach similar final val_loss at this scale (RMSNorm
drops mean-centering/bias, which theory says barely matters for transformers) — the interesting
comparison is really compute cost, not quality, at S-tier.

**Observation:** LayerNorm finished at val_loss 3.4878 vs baseline's 3.5037 (delta -0.0158),
consistently ~0.015-0.027 ahead at every logged checkpoint from step 200 onward — not just
noise at one point, but the margin also isn't growing, it hovers right around the seed-noise
floor (0.0150, D-035) the whole second half of training.

**Conclusion:** a **borderline** result — real in the sense that the trend is consistent, but the
final-step delta (-0.0158) is right at the noise-floor boundary, so this shouldn't be
oversold as "LayerNorm beats RMSNorm here." Given RMSNorm's mean-centering/bias overhead is a
known small compute+memory cost (extra params: LayerNorm's model here is 9.72M vs RMSNorm's
9.71M, negligible at this scale but scales with `d_model` at larger sizes), RMSNorm's
quality-per-compute is still the better default — matches the literature's framing (RMSNorm
"matches" rather than "beats" LayerNorm).
