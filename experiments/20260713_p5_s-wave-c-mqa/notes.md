# 20260713_p5_s-wave-c-mqa — MQA, 1 KV head

Single variable vs the Wave C MHA control: `n_kv_heads` 4 → 1 (Shazeer '19). All 4 query heads
share ONE KV head.

- **Result:** val_loss **3.5498** (ppl 34.81) — **+0.0186 vs MHA control**, just past the noise
  floor. Real, and the **only** variant that clearly *loses* quality — the price of collapsing to
  a single shared KV head (no per-head K/V diversity left).
- **KV cache:** 256 B/token/layer — **4× smaller** than MHA, the smallest of the wave.
- **params:** 9.34M (smallest).

Takeaway: MQA is the quality-vs-cache extreme — cheapest cache, but the first to cost you quality.
MLA (below) reaches a nearly-as-small cache *without* the quality hit.
