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

## Wave C — Attention variants (2026-07-13)

Control: `20260713_p5_s-wave-c-mha` — plain MHA at **n_heads=4** (NOT the 3-head
`p4_s_baseline`; GQA-2 is undefined at 3 heads, so the whole wave runs at 4 heads). Noise floor:
0.0150 (D-035). Figures: `docs/results/wave_c_attention_variants.png` (curves + cache-vs-quality
tradeoff), cache/tok-s data in `docs/results/wave_c_inference_bench.csv`, mechanism walkthrough in
`notebooks/06_mla_explained.ipynb`. New code: `MLAAttention` (DeepSeek-V2 §2), incremental
KV-cache decode for all 4 variants (`kv_cache.py`, rewritten `generate()`), `bench_inference.py`.
- **Quality is nearly flat across all four** (val 3.5107–3.5498, spread 0.039 ≈ 2.6× the noise
  floor): at S-tier/98M tokens the attention *type* barely moves loss. So judge on **cache**.
- **GQA-2:** −0.0205 vs MHA (real, marginally better) at **2× smaller** cache (512 vs 1024
  B/tok/layer). Halving KV heads is free on quality — why GQA is the industry default.
- **MQA:** +0.0186 vs MHA (real, the ONLY quality loss) but **4× smaller** cache (256 B). The
  quality-vs-cache extreme: cheapest cache, first to cost quality.
- **MLA:** −0.0166 vs MHA (real, marginally better) at **3.2× smaller** cache (320 B) — smallest
  cache among the quality-*preserving* variants. Reproduces DeepSeek-V2's headline at 10M params:
  near-MQA cache, near-MHA quality, by rebuilding all 4 content heads from a low-rank latent +
  decoupled RoPE key. Trade: ~25% slower decode/token at this scale (extra projections, no
  absorption trick) — cache win is memory, not single-stream latency here.
- **Verdict for phase 9's recipe:** MHA's full cache buys nothing at this scale. Default to
  **GQA** for simplicity (2× cache cut, zero quality cost, trivial code), and reach for **MLA**
  when KV-cache memory is the binding constraint (long context / large batch), accepting its
  decode-compute overhead (which absorption + a pre-allocated cache would remove). MQA only if
  cache is the single overriding constraint and a small quality hit is acceptable.
