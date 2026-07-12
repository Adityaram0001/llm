#!/usr/bin/env python
"""Pre-run batch-size calibration (D-018): sweep micro-batch on the CURRENT hardware for a
given train config's model + seq_len, doubling until OOM or a tokens/sec plateau (~2 min).
Report the sweet spot + memory at each point.

This is a one-time-per-hardware calibration tool, **not a runtime controller** -- effective
batch stays FIXED in the training config; `micro_batch x grad_accum` re-factorizes it per
machine. Never auto-adjust batch size mid-run: it's a hyperparameter, and drifting it kills
run-to-run comparability (D-018). Re-run this once per new hardware (Mac, a rented 5090, ...)
before launching real training there.

Usage:
    python scripts/find_batch_size.py --config configs/train_s_baseline.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from llmlab.model import GPT, ModelConfig
from llmlab.train.config import TrainConfig
from llmlab.utils import autocast_ctx, get_device, mem_stats

ROOT = Path(__file__).resolve().parents[1]


def bench_micro_batch(model: GPT, seq_len: int, micro_batch: int, device: torch.device, n_iters: int = 5) -> tuple[float, float]:
    """(tokens/sec, memory GB) for `n_iters` fwd+bwd steps at this micro-batch. Uses random
    tokens -- this measures compute/memory, not learning, so no real data is needed."""
    x = torch.randint(0, model.cfg.vocab_size, (micro_batch, seq_len), device=device)
    y = torch.randint(0, model.cfg.vocab_size, (micro_batch, seq_len), device=device)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            _, loss = model(x, y)
        loss.backward()

    for _ in range(2):  # warmup, not timed
        step()
    _sync(device)

    t0 = time.time()
    for _ in range(n_iters):
        step()
    _sync(device)
    elapsed = time.time() - t0

    tokens_per_sec = (micro_batch * seq_len * n_iters) / elapsed
    mem_gb = mem_stats().get(f"{device.type}_allocated_mb", mem_stats()["rss_mb"]) / 1024
    return tokens_per_sec, mem_gb


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="a configs/train_*.yaml (reads model_config + seq_len)")
    parser.add_argument("--max-micro-batch", type=int, default=256)
    parser.add_argument("--plateau-tolerance", type=float, default=0.05, help="stop doubling once tok/s gains less than this fraction over the running best")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device(cfg.device) if cfg.device else get_device()
    model_cfg = ModelConfig.from_yaml(str(ROOT / cfg.model_config))
    model = GPT(model_cfg).to(device)
    print(f"device={device}  model={cfg.model_config}  params={model.num_params() / 1e6:.2f}M  seq_len={cfg.seq_len}\n")

    results: list[tuple[int, float, float]] = []
    best_tps = 0.0
    mb = 1
    while mb <= args.max_micro_batch:
        try:
            tps, mem_gb = bench_micro_batch(model, cfg.seq_len, mb, device)
        except RuntimeError as e:
            print(f"micro_batch={mb:>4}  OOM ({e})")
            break
        print(f"micro_batch={mb:>4}  tokens/sec={tps:>9,.0f}  mem={mem_gb:.2f}GB")
        # bool(...) matters: `results and ...` would otherwise alias the mutable `results`
        # list itself when falsy, which the very next line's `.append()` then mutates too.
        plateaued = bool(results) and tps < best_tps * (1 + args.plateau_tolerance)
        results.append((mb, tps, mem_gb))
        if plateaued:
            print(f"  -> plateaued (< {args.plateau_tolerance:.0%} gain over the running best) -- stopping sweep")
            break
        best_tps = max(best_tps, tps)
        mb *= 2

    mb, tps, mem_gb = max(results, key=lambda r: r[1])
    print(
        f"\nsweet spot: micro_batch={mb} ({tps:,.0f} tok/s, {mem_gb:.2f}GB). Set "
        f"batch.micro_batch={mb} in the train config and batch.grad_accum = "
        f"effective_batch_tokens // ({mb} * {cfg.seq_len})."
    )


if __name__ == "__main__":
    main()
