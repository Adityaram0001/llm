# Phase 3 — Model architecture (config-driven GPT)

**Goal:** a clean decoder-only transformer in `src/llmlab/model/` where every technique we'll
study is a config switch, plus S/M/L size configs and a tensor-shape walkthrough notebook.
**Effort:** 1–2 sessions. Second flagship learning phase.

## Design (per D-002)

**`src/llmlab/model/config.py`** — `ModelConfig` dataclass (loaded from YAML):

```python
vocab_size, d_model, n_layers, n_heads, n_kv_heads,  # n_kv_heads: =n_heads→MHA, 1→MQA, else GQA
head_dim, max_seq_len, dropout,
norm: "layernorm"|"rmsnorm", norm_position: "pre"|"post", qk_norm: bool,
pos_encoding: "learned"|"sinusoidal"|"rope"|"alibi"|"none", rope_theta,
ffn: "gelu"|"swiglu", ffn_mult,
attention: "mha_gqa"|"mla",  mla: {kv_lora_rank, q_lora_rank, rope_head_dim},  # phase 5-C
tie_embeddings: bool, init: "gpt2"|"scaled",
moe: null | {n_experts, n_shared, top_k, balancing: "aux_loss"|"bias_free"},   # phase 5-F
mtp: null | {n_predict_tokens},                                               # phase 5-F
```

**Files:** `model/attention.py`, `model/ffn.py`, `model/norms.py`, `model/positional.py`,
`model/block.py`, `model/gpt.py` (embeddings→blocks→final norm→head; `forward(idx, targets)`
returns logits, loss; `generate()` with temperature/top-k/top-p; `num_params()`;
`estimate_flops_per_token()`).

Phase 3 implements the **baseline path** fully: MHA/GQA via `F.scaled_dot_product_attention`,
learned+RoPE positions, both norms, both FFNs, weight tying. MLA/MoE/MTP raise
`NotImplementedError` with a pointer to phase 5 (but config fields exist NOW so old configs
stay loadable).

## Size configs (create `configs/model_{s,m,l}.yaml`; verify param counts with a table)

| Tier | d_model | layers | heads | vocab | ~params (tied) |
|------|---------|--------|-------|-------|----------------|
| S | 384 | 6 | 6 | 16k | ~10M |
| M | 512 | 10 | 8 | 16k | ~35M |
| L | 768 | 12 | 12 | 16k | ~105M |

Implementer: compute exact counts, print a per-component breakdown (embed/attn/ffn/head %),
adjust to hit tiers, put the table in the shape notebook AND in configs as comments.

## Deliverables

1. The model package, importable, unit-tested: **`tests/test_model.py`** — shapes; causal mask
   (future token change must NOT affect past logits — test this explicitly!); loss ≈ ln(vocab)
   at init; generate() runs; every config combo instantiates (parametrized test); tied weights
   share storage; RoPE relative-shift property.
2. **`notebooks/04_shapes_walkthrough.ipynb`**: one forward pass narrated tensor-by-tensor with
   shapes at every step; attention-weight heatmap of the untrained model; param-budget pie.
   (This is the "every step explained" artifact — invest here.)
3. Overfit-one-batch sanity: script or notebook cell that drives loss →~0 on a single batch
   (the classic "your model can learn" test).

## Decision points

- Weight tying on/off for baseline (recommend ON at these scales — log why with the math).
- head_dim 64 fixed vs derived; dropout 0.0 vs 0.1 for pretraining (recommend 0.0, cite modern practice).
- Init scheme: GPT-2 (0.02, scaled residual) — explain each term.

## Learning checkpoints

- Draw the block diagram from memory; state every weight matrix's shape at S-tier.
- Why is loss ≈ ln(vocab_size) at init? Why scale residual init by 1/√(2·n_layers)?
- Where do params live (embedding vs attn vs FFN) and how does that shift with vocab/d_model?

## Exit criteria
Tests green on mps AND cpu; notebook complete; configs committed; decisions logged; PROGRESS updated.
