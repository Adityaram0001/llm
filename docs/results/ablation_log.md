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

## Wave D — Optimizers & schedules (2026-07-13)

Control: `20260713_p5_s-wave-d-control` — Wave D's own control (mb=64/accum=2 GPU-tuned recipe,
same 65,536 tok/step effective batch as `p4_s_baseline`, val 3.4977, within noise of it). Noise
floor: 0.015-0.02 (D-035). Figure: `docs/results/wave_d_optimizers_schedules.png`. New code:
`Muon`/`Lion` optimizers + Newton-Schulz orthogonalization (`train/optimizers.py`), hybrid
Muon+AdamW optimizer construction/checkpointing, `cosine`/`wsd`/`constant` schedule dispatch,
PaLM z-loss (all in `train/{config,trainer}.py`).
- **Muon:** **best of the wave** (-0.1545 vs AdamW control, >10x noise floor) — gap largest early,
  narrowing but never closing; reproduces the nanoGPT speedrun's "faster convergence" claim.
- **Lion:** worst of the wave (+0.4226) but a one-shot, un-tuned lr/wd conversion from the
  AdamW recipe — flagged as needing a real sweep, not a fair verdict against Lion itself.
- **WSD vs cosine:** real win (-0.1213) — already ahead before its own decay phase even starts,
  showing cosine's continuous early decay costs real ground.
- **constant (no decay) vs cosine:** real win (-0.0674), establishing the hierarchy
  **WSD > constant > cosine** — *when* you decay matters as much as whether you decay at all.
- **WSD multi-budget bonus:** two decay forks off `wave_d_constant`'s SAME shared step-1500
  checkpoint (+10%/+26.7% tokens) reach 3.3220/3.2768 — a clean demonstration that WSD lets you
  decide the final token budget after training, not before.
- **z-loss, AdamW wd=0/beta2=0.999:** all null results (within noise) at this token budget — no
  failure mode for z-loss to fix yet, and not enough training for wd/beta2's long-horizon
  mechanisms to differentiate.
- **grad-clip off:** real but undramatic (+0.0215) — no spike (`clip_grad_norm_` logs the
  pre-clip norm regardless of whether it applies, so the metric can't show a difference by
  construction), just steady small degradation; this depth/warmup combo is already stable enough
  that clipping rarely binds.
- **batch-size study (fixed ~98.3M tokens, lr NOT rescaled):** monotonically worse with fewer/
  bigger-batch steps (control 3.50 → 0.25M-batch 4.26 → 1M-batch 5.39) — confirms the batch/
  steps/lr coupling, though the 1M point is partly confounded by an unscaled 30-step warmup
  eating 32% of its 94-step budget.
- **Verdict for phase 9's recipe:** **Muon** is the strongest single lever found in the project
  so far if training speed/compute is the binding constraint; **WSD** (or at minimum, decay only
  late rather than continuously) beats cosine for free; the AdamW wd/beta2 defaults (D-021) stand
  unchanged; z-loss and grad-clip are cheap insurance worth keeping even though this scale can't
  prove their value yet.
