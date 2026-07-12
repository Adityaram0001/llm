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

## Wave B — Positional encodings (2026-07-12)

Baseline: `20260711_p4_s-baseline` (RoPE). Noise floor: 0.0150 (D-035). RW-5's `GPT.forward()`
fix (allow eval past `max_seq_len` for rope/alibi/none) landed this wave, unblocking the
length-extrapolation probe. Figure: `docs/results/wave_b_positional_encodings.png`.
- **learned:** real, worse (+0.227 val_loss vs RoPE) — cannot extrapolate past 512 by
  construction (fixed table), confirmed via `ValueError`.
- **sinusoidal:** real, WORST of the wave (+1.486) — notably worse than even `learned`, a
  surprise (losing the learnable position parameters costs far more than expected at this
  scale). Also cannot extrapolate past 512.
- **ALiBi:** real, BEST of the wave (-0.021 at trained length) AND the standout result —
  val_loss *improves* with more context (ppl 32.56->32.08->31.67 at 512->1024->2048), a clean
  small-scale reproduction of the paper's headline length-extrapolation claim.
- **NoPE:** real, worse at trained length (+0.196) AND catastrophic under extrapolation (ppl
  40->67->732 at 512->1024->2048) — the sharpest contrast in the probe, opposite end from ALiBi.
- **Verdict for phase 9's recipe:** RoPE (current default) is solid but **ALiBi is a genuine
  contender**, especially for any future long-context need (RW-5's phase-9 capstone chat-context
  goal) — its extrapolation behavior is strictly better than RoPE's at this scale. Learned/
  sinusoidal/NoPE are all worse choices for this project, ruled out.
