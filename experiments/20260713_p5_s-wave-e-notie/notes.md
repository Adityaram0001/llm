# 20260713_p5_s-wave-e-notie — weight tying off

**Hypothesis:** D-016 tied the input embedding and output unembedding matrix for parameter-budget
reasons ("same quality at lower param cost"), not because untying was expected to actively hurt.
This run tests whether untying (same d_model/layers/heads, +3.07M params for a separate
unembedding matrix, 9.71M -> 12.79M total) changes quality at S-tier.

- **Result:** val_loss 3.4699 (ppl 32.13), **-0.0278 vs control (3.4977)** — just past the
  0.015-0.02 noise floor (D-035). REAL, but with an important caveat below.
- **Caveat — this is NOT a param-matched comparison.** The untied model has 31.6% more total
  parameters than the control (12.79M vs 9.71M). The observed win could simply be extra model
  capacity from the unshared unembedding matrix, not evidence that tying itself costs quality at
  a FIXED layer shape. A fair ceteris-paribus test would need a param-matched untied variant
  (e.g. shrink d_model or layers to compensate for the extra ~3.07M embedding params) — Wave E
  didn't budget for that second run.
- **Conclusion:** this result does not overturn D-016's parameter-budget argument (tying still
  buys the same active-compute quality at lower total param cost, which is what D-016 actually
  claimed) but it also does NOT show tying is quality-neutral at a fixed layer shape — that
  claim needs the param-matched follow-up before treating it as settled. Flag as a candidate for
  Wave G or a phase-9 recipe footnote if the extra rigor is ever worth the run.
