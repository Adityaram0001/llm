# Phase 0 — Environment & MPS baseline

**Goal:** working venv, `llmlab` package skeleton, proof that Apple-GPU training works, and
**measured** throughput numbers that calibrate every later time estimate.
**Effort:** one short session.

## Deliverables

1. **`scripts/setup.sh`** (exists — review, then the USER runs it in terminal): creates `.venv`,
   installs `requirements.txt`, `pip install -e .`, registers Jupyter kernel `llm-lab`.
2. **`pyproject.toml`** (exists — verify editable install works).
3. **`src/llmlab/utils.py`**: `set_seed(seed)` (python/numpy/torch incl. mps), `get_device()`
   (mps → cpu fallback), `param_count(model)`, `mem_stats()` (psutil RSS + torch.mps allocated).
4. **`scripts/verify_env.py`**: prints python/torch versions; asserts `torch.backends.mps.is_available()`;
   runs a bf16-autocast matmul on mps; checks `F.scaled_dot_product_attention` on mps; verifies
   wandb import (offline mode OK); exits 0 with a green summary.
5. **`scripts/bench_mps.py`**: (a) matmul benchmark → effective TFLOPS at fp32/bf16 for sizes
   1k–4k; (b) build a throwaway ~10M-param GPT (use a minimal inline model; phase 3 replaces it)
   and measure fwd+bwd **tokens/sec** at seq 256/512/1024, micro-batch sweep until memory
   complains. CPU-vs-MPS comparison at one setting (teachable moment).
6. **`notebooks/00_mps_playground.ipynb`**: guided tour — tensors on mps, autocast dtypes,
   timing pitfalls (`torch.mps.synchronize()` before timing!), memory readout.
7. Record measured numbers in DECISIONS.md (D-008: "compute budget") — include estimated
   wall-clock for: 10M model × 100M tokens; 100M model × 1B tokens.

## Decision points (present options, record in DECISIONS.md)

- Exact torch version (latest stable) — note MPS fixes in release notes.
- wandb online vs offline default.

## Gotchas for the implementer

- Time GPU ops only after `torch.mps.synchronize()`; warm up before measuring.
- fp16 on MPS can produce NaNs in softmax — prefer bf16 everywhere.
- Don't ship `torch.compile` in the bench default path; optional `--compile` flag is fine.

## Learning checkpoints (user should be able to answer)

- Why bf16 rather than fp16? (exponent bits vs mantissa; loss-scale-free training)
- What does "tokens/sec" depend on — and why does micro-batch size change it?
- Where do the 4 memory consumers live (weights/grads/optimizer/activations) and which
  dominates at 10M vs 100M params?

## Exit criteria
All PROGRESS.md phase-0 boxes checked; PROGRESS.md updated; D-008 logged.
