#!/usr/bin/env python
"""Wave E (phase 5): measure PEAK memory of one fwd+bwd step across a sweep of seq_len, with and
without gradient checkpointing, at a fixed micro_batch. Activation memory (not parameter/
optimizer memory) is what scales with seq_len, so this isolates that curve -- pairs with
`find_batch_size.py`'s micro-batch sweep (which fixes seq_len and varies batch instead) rather
than replacing it.

Uses random tokens -- this measures compute/memory, not learning, so no real data is needed
(same pattern as `find_batch_size.py`'s `bench_micro_batch`).

Usage:
    python scripts/bench_activation_memory.py --config configs/train_s_wave_e_gradckpt.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from llmlab.model import GPT, ModelConfig
from llmlab.train.config import TrainConfig
from llmlab.utils import autocast_ctx, get_device

ROOT = Path(__file__).resolve().parents[1]


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    if device.type == "mps":
        # MPS has no reset-able peak counter -- current_allocated right after a step is the
        # best available proxy (activations have been freed by then except transient overlap).
        return torch.mps.current_allocated_memory() / (1024**2)
    import psutil
    import os

    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def bench_seq_len(model: GPT, micro_batch: int, seq_len: int, device: torch.device) -> float:
    x = torch.randint(0, model.cfg.vocab_size, (micro_batch, seq_len), device=device)
    y = torch.randint(0, model.cfg.vocab_size, (micro_batch, seq_len), device=device)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            _, loss = model(x, y)
        loss.backward()

    step()  # warmup (not measured -- allocator caching would otherwise understate the first point)
    _sync(device)
    _reset_peak(device)
    step()
    _sync(device)
    return _peak_mb(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="a configs/train_*.yaml (reads model_config + batch.micro_batch)")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--micro-batch", type=int, default=None, help="override the config's batch.micro_batch")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="also toggle model.gradient_checkpointing on for this sweep")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path (defaults to docs/results/wave_e_activation_memory<_gradckpt>.csv)")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device(cfg.device) if cfg.device else get_device()
    model_cfg = ModelConfig.from_yaml(str(ROOT / cfg.model_config))
    model = GPT(model_cfg).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    mb = args.micro_batch or cfg.batch.micro_batch

    print(
        f"device={device}  model={cfg.model_config}  params={model.num_params() / 1e6:.2f}M  "
        f"micro_batch={mb}  gradient_checkpointing={args.gradient_checkpointing}\n"
    )

    rows = []
    for seq_len in args.seq_lens:
        try:
            peak_mb = bench_seq_len(model, mb, seq_len, device)
        except RuntimeError as e:
            print(f"seq_len={seq_len:>5}  OOM ({e})")
            break
        print(f"seq_len={seq_len:>5}  peak_mem={peak_mb:>9,.1f} MB")
        rows.append((seq_len, mb, args.gradient_checkpointing, peak_mb))

    out = args.out or ROOT / "docs" / "results" / (
        f"wave_e_activation_memory{'_gradckpt' if args.gradient_checkpointing else ''}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "micro_batch", "gradient_checkpointing", "peak_mem_mb"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
