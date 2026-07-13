# 20260713_p5_s-wave-c-gqa2 — GQA, 2 KV groups

Single variable vs the Wave C MHA control: `n_kv_heads` 4 → 2 (Ainslie et al. '23). 4 query heads
share 2 KV heads.

- **Result:** val_loss **3.5107** (ppl 33.47) — **−0.0205 vs MHA control** (3.5312), just past the
  0.015 seed-noise floor (D-035). Real, marginally *better*, and the best val of the wave.
- **KV cache:** 512 B/token/layer — **half** MHA's, for no quality cost. This is exactly why GQA
  is the industry default (LLaMA-2-70B, Mistral): most of MHA's cache is redundant across heads.
- **params:** 9.71M (smaller K/V projections than MHA).

Takeaway: at S-tier/98M tokens, halving the KV heads is free on quality and halves the cache.
