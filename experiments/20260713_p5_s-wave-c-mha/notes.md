# 20260713_p5_s-wave-c-mha — Wave C control (plain MHA, n_heads=4)

**Role:** the reference run for Wave C. NOT the 3-head `20260711_p4_s-baseline` — GQA "2 groups"
is undefined at `n_heads=3` (3 is not divisible by 2), so the whole wave runs at `n_heads=4,
head_dim=64` (attn inner dim 256, o_proj 256→192) and this MHA-4 run is the internal control.
Every gqa2/mqa/mla delta is measured against this. See D-038.

- **Result:** val_loss **3.5312** (ppl 34.16), 1500 steps / 98.3M tokens, lr 1e-3, seed 1337,
  micro_batch 16 × grad_accum 8 (effective 65,536 tok), identical harness to Waves A/B.
- **KV cache:** 1024 B/token/layer (bf16) = `2 · n_kv_heads(4) · head_dim(64) · 2B`. The largest
  of the wave — every other variant shrinks this (see `docs/results/wave_c_inference_bench.csv`).
- **params:** 10.45M (larger than the 9.71M 3-head baseline purely from the 4th head).

Samples pick up the corpus's Socratic-dialogue register as expected. This run only exists to make
the other three single-variable comparisons valid.
