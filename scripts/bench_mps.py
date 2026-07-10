#!/usr/bin/env python
"""Benchmark Apple-GPU (MPS) throughput to calibrate later time estimates.

(a) Raw matmul TFLOPS at fp32 / bf16 for square matrices 1k-4k.
(b) A throwaway ~10M-param GPT (phase 3 replaces this with the real model):
    measure fwd+bwd tokens/sec at seq_len 256/512/1024, sweeping micro-batch
    size until MPS complains about memory. One CPU-vs-MPS comparison point.

Usage: python scripts/bench_mps.py [--compile]

Run from a terminal, not a notebook (this is a long-running benchmark script
per the project's hardware rules).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llmlab.utils import get_device, mem_stats, param_count, set_seed  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_matmul(device: torch.device, sizes: list[int], dtype: torch.dtype, reps: int = 20) -> dict[int, float]:
    """Return {size: effective TFLOPS} for square matmuls of the given dtype."""
    results = {}
    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)

        # warm up
        for _ in range(3):
            c = a @ b
        sync(device)

        start = time.perf_counter()
        for _ in range(reps):
            c = a @ b
        sync(device)
        elapsed = time.perf_counter() - start

        flops = 2 * (n**3) * reps  # multiply-add = 2 flops
        tflops = flops / elapsed / 1e12
        results[n] = tflops
        del a, b, c
    return results


class CausalSelfAttention(nn.Module):
    """Manual QKV + F.scaled_dot_product_attention, so the benchmark exercises
    the same fused-attention path the real model (phase 3) will use."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.c_proj(out)


class TinyGPT(nn.Module):
    """Minimal decoder-only transformer for throughput benchmarking only.
    Phase 3 replaces this with the real, config-driven model."""

    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn": CausalSelfAttention(d_model, n_head),
                        "ln2": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Linear(4 * d_model, d_model),
                        ),
                    }
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = x + blk["attn"](blk["ln1"](x))
            x = x + blk["mlp"](blk["ln2"](x))
        x = self.ln_f(x)
        return self.head(x)


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def bench_gpt_throughput(
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    seq_len: int,
    batch_sizes: list[int],
    reps: int = 5,
) -> list[dict]:
    rows = []
    for bs in batch_sizes:
        try:
            x = torch.randint(0, vocab_size, (bs, seq_len), device=device)
            y = torch.randint(0, vocab_size, (bs, seq_len), device=device)

            # warm up
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync(device)

            start = time.perf_counter()
            for _ in range(reps):
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("mps", "cuda")
                ):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                model.zero_grad(set_to_none=True)
            sync(device)
            elapsed = time.perf_counter() - start

            tokens = bs * seq_len * reps
            toks_per_sec = tokens / elapsed
            mem = mem_stats()
            rows.append(
                {
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "tokens_per_sec": toks_per_sec,
                    "rss_mb": mem["rss_mb"],
                    "mps_allocated_mb": mem.get("mps_allocated_mb"),
                }
            )
            print(
                f"    bs={bs:4d} seq={seq_len:5d}  "
                f"{toks_per_sec:9.0f} tok/s  rss={mem['rss_mb']:.0f}MB "
                f"mps_alloc={mem.get('mps_allocated_mb', 0):.0f}MB"
            )
            del x, y, logits, loss
        except RuntimeError as e:
            if is_oom(e):
                print(f"    bs={bs:4d} seq={seq_len:5d}  OOM — stopping sweep for this seq_len")
                model.zero_grad(set_to_none=True)
                break
            raise
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="try torch.compile (unreliable on MPS, optional)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")

    print("=== (a) Matmul TFLOPS ===")
    sizes = [1024, 2048, 4096]
    for dtype in (torch.float32, torch.bfloat16):
        print(f"\n-- dtype={dtype} --")
        results = bench_matmul(device, sizes, dtype)
        for n, tflops in results.items():
            print(f"  {n:5d}x{n:<5d}  {tflops:6.2f} TFLOPS")

    print("\n=== (b) TinyGPT (~10M param) tokens/sec ===")
    vocab_size = 16000
    model = TinyGPT(vocab_size=vocab_size, d_model=256, n_layer=6, n_head=4, block_size=1024).to(device)
    n_params = param_count(model)
    print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)\n")

    if args.compile:
        print("Compiling model (torch.compile — optional, unreliable on MPS)...")
        model = torch.compile(model)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    all_rows = []
    for seq_len in (256, 512, 1024):
        print(f"\n  seq_len={seq_len}")
        rows = bench_gpt_throughput(model, device, vocab_size, seq_len, batch_sizes)
        all_rows.extend(rows)

    print("\n=== CPU vs MPS comparison (bs=4, seq=256) ===")
    cpu_model = TinyGPT(vocab_size=vocab_size, d_model=256, n_layer=6, n_head=4, block_size=1024).to("cpu")
    cpu_model.load_state_dict(
        {k: v.cpu() for k, v in (model._orig_mod if args.compile else model).state_dict().items()}
    )
    print("  MPS:")
    bench_gpt_throughput(model, device, vocab_size, 256, [4], reps=5)
    print("  CPU:")
    bench_gpt_throughput(cpu_model, torch.device("cpu"), vocab_size, 256, [4], reps=5)

    print("\nDone. Copy the numbers you want into docs/DECISIONS.md (D-008).")


if __name__ == "__main__":
    main()
