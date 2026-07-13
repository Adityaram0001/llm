# 20260713_p5_s-wave-c-mla — Multi-head Latent Attention (DeepSeek-V2 §2)

Single variable vs the Wave C MHA control: `attention` mha_gqa → mla. Head-dim-preserving sizing
(kv_lora_rank 128, q_lora_rank 192, nope 32, rope 32, v 64 → per-head Q/K = 64 = MHA's head_dim).
Implementation + matrix diagrams: `src/llmlab/model/attention.py::MLAAttention`,
`notebooks/06_mla_explained.ipynb`.

- **Result:** val_loss **3.5146** (ppl 33.60) — **−0.0166 vs MHA control**, just past the noise
  floor. Real, marginally *better* than MHA, statistically tied with GQA-2.
- **KV cache:** 320 B/token/layer = `(kv_lora_rank 128 + rope_head_dim 32) · 2B` — **3.2× smaller
  than MHA**, and the smallest among the quality-*preserving* variants (MQA is smaller, 256 B, but
  real-worse). This is the DeepSeek-V2 headline reproduced at 10M params: near-MQA cache with
  near-MHA quality, because MLA keeps all 4 content heads (rebuilt from the latent) instead of
  collapsing them.
- **The trade (measured, honest):** decode ~85 tok/s vs MHA's ~116 at S-tier — ~25% slower/token,
  because our `MLAAttention` re-expands per-head K/V from the latent every step (extra down/up
  matmuls) and skips the production "weight absorption" trick. At 10M params single-stream decode
  is launch-overhead-bound anyway, so the cache's real payoff here is **memory** (→ larger batch /
  longer context), not latency. See `docs/results/wave_c_inference_bench.csv` and notebook §4.
- **params:** 10.73M (largest — the low-rank projections add params at this tiny scale).

Takeaway: MLA is the Pareto-interesting point — it dominates MHA (smaller cache, equal quality)
and matches GQA's quality at a smaller cache, paying in decode compute that a real serving stack
(absorption + pre-allocated cache) would largely remove.
