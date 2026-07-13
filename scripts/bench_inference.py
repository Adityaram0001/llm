#!/usr/bin/env python
"""Wave C inference bench: KV-cache bytes/token (analytical + empirical) and decode tok/s.

The training runs answer "does the cheaper-cache attention variant keep quality?"; this script
answers the other half — "how much cache does it actually save, and how fast does it decode?".
For each attention config it:
  1. prints the analytical KV-cache bytes/token/layer (the paper's cache-table number),
  2. verifies it empirically by building a real cache and reading its `.nbytes()`,
  3. times autoregressive decode (prefill P tokens, then generate N with the KV cache) at several
     context lengths, plus the no-cache O(T)-per-step fallback for contrast.

Device-agnostic (CLOUD.md): device via get_device(), sync guarded by is_available(). Weights are
random — quality is not measured here, only memory and speed, which don't depend on training.

Usage:  python scripts/bench_inference.py [--seq-lens 512 1024 2048] [--new-tokens 128] [--out CSV]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from llmlab.model import GPT, ModelConfig
from llmlab.model.attention import make_cache
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "mha": "configs/model_s_attn_mha.yaml",
    "gqa2": "configs/model_s_attn_gqa2.yaml",
    "mqa": "configs/model_s_attn_mqa.yaml",
    "mla": "configs/model_s_attn_mla.yaml",
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def analytical_bytes_per_token(cfg: ModelConfig, dtype_size: int = 2) -> int:
    """Cache bytes per token PER LAYER, straight from the config (no model needed)."""
    if cfg.attention == "mla":
        return (cfg.mla.kv_lora_rank + cfg.mla.rope_head_dim) * dtype_size
    return 2 * cfg.n_kv_heads * cfg.head_dim * dtype_size  # K and V at n_kv_heads


@torch.no_grad()
def decode_tokens_per_sec(
    model: GPT, cfg: ModelConfig, prompt_len: int, new_tokens: int, device: torch.device,
    use_cache: bool,
) -> float:
    """Prefill `prompt_len` tokens then generate `new_tokens`, return decode tokens/sec."""
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    # warmup (kernels / autotune)
    _ = model.generate(prompt, 4, top_k=1, use_cache=use_cache)
    sync(device)
    t0 = time.perf_counter()
    _ = model.generate(prompt, new_tokens, top_k=1, use_cache=use_cache)
    sync(device)
    return new_tokens / (time.perf_counter() - t0)


@torch.no_grad()
def empirical_cache_bytes_per_token(
    model: GPT, cfg: ModelConfig, seq_len: int, device: torch.device
) -> float:
    """Build a real per-layer cache by decoding `seq_len` tokens, then read total bytes / token."""
    caches = [make_cache(cfg) for _ in range(cfg.n_layers)]
    idx = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
    model(idx, caches=caches)
    for _ in range(seq_len - 1):
        idx = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
        model(idx, caches=caches)
    total = sum(c.nbytes() for c in caches)
    return total / seq_len  # bytes/token across ALL layers (batch=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/results/wave_c_inference_bench.csv")
    args = ap.parse_args()

    device = get_device()
    # bf16 is the real inference dtype (2 bytes/value) -- matching it makes the empirical cache
    # size validate the analytical bf16 number directly rather than coming out 2x (fp32).
    print(f"device: {device} (inference dtype: bf16)\n")

    rows = []
    # -- cache bytes/token (analytical vs empirical, all layers) --
    print(f"{'variant':6s} {'analytic B/tok/layer':>22s} {'x n_layers (KiB)':>18s} {'empirical B/tok (all layers)':>30s}")
    for name, path in VARIANTS.items():
        cfg = ModelConfig.from_yaml(str(ROOT / path))
        model = GPT(cfg).to(device).to(torch.bfloat16)
        a = analytical_bytes_per_token(cfg)
        a_all_kib = a * cfg.n_layers / 1024
        emp = empirical_cache_bytes_per_token(model, cfg, 256, device)
        print(f"{name:6s} {a:>22d} {a_all_kib:>18.1f} {emp:>30.1f}")
        rows.append({
            "variant": name, "metric": "cache_bytes_per_token_per_layer", "seq_len": "", "value": a,
        })
        rows.append({
            "variant": name, "metric": "cache_bytes_per_token_all_layers_empirical",
            "seq_len": 256, "value": round(emp, 1),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- decode tok/s at each context length, cached vs full-recompute --
    print(f"\n{'variant':6s} {'ctx':>6s} {'cached tok/s':>13s} {'nocache tok/s':>14s} {'speedup':>8s}")
    for name, path in VARIANTS.items():
        cfg = ModelConfig.from_yaml(str(ROOT / path))
        model = GPT(cfg).to(device).to(torch.bfloat16)
        for P in args.seq_lens:
            cached = decode_tokens_per_sec(model, cfg, P, args.new_tokens, device, use_cache=True)
            nocache = decode_tokens_per_sec(model, cfg, P, args.new_tokens, device, use_cache=False)
            print(f"{name:6s} {P:>6d} {cached:>13.1f} {nocache:>14.1f} {cached / nocache:>7.2f}x")
            rows.append({"variant": name, "metric": "decode_tok_s_cached", "seq_len": P, "value": round(cached, 1)})
            rows.append({"variant": name, "metric": "decode_tok_s_nocache", "seq_len": P, "value": round(nocache, 1)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "metric", "seq_len", "value"])
        w.writeheader()
        w.writerows(rows)
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
