# Ablation log — 5-line summaries per wave (full detail in each run's notes.md + registry.csv)

## Wave A — Norms & activations (2026-07-12)

Baseline: `20260711_p4_s-baseline` (rmsnorm/pre-norm/swiglu/no-qk-norm). Noise floor: 0.0150
(D-035). Figure: `docs/results/wave_a_norms_activations.png`.
- **RMSNorm vs LayerNorm:** borderline (-0.0158, right at the noise floor) — RMSNorm kept as
  default for its lower compute cost, not because LayerNorm was beaten decisively.
- **pre-norm vs post-norm:** post-norm stagnates near loss~6.8 by step 150 and never recovers
  (not a blow-up — grad_norm stayed <=1.52) — confirms pre-norm's necessity at this depth.
- **SwiGLU vs GELU (param-matched):** SwiGLU wins clearly and consistently, +0.17-0.2 val_loss
  gap from step 200 onward — validates D-016's SwiGLU default.
- **+QK-norm:** best result of the wave, -0.062 val_loss, gap widening over training — a real,
  robust win even at S-tier/shallow depth, contrary to the "matters mainly at scale" expectation.
- **Verdict for phase 9's recipe:** carry forward rmsnorm + pre-norm + swiglu (all already
  defaults) + **add qk_norm=true** as a new recommended default; LayerNorm not worth switching to.
