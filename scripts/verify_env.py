#!/usr/bin/env python
"""Sanity-check the training environment: MPS availability, bf16 autocast,
scaled_dot_product_attention on MPS, and wandb import. Exits 0 with a green
summary if everything works.

Usage: python scripts/verify_env.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llmlab.utils import get_device, mem_stats, set_seed  # noqa: E402

CHECK = "\033[32m✓\033[0m"
CROSS = "\033[31m✗\033[0m"


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    print(f"Python:  {sys.version.split()[0]}")
    print(f"Torch:   {torch.__version__}")

    mps_available = torch.backends.mps.is_available()
    results.append(("MPS available", mps_available, str(mps_available)))
    if not mps_available:
        print(f"{CROSS} MPS not available — falling back to CPU for the rest of this check.")

    device = get_device()
    print(f"Device:  {device}")

    set_seed(42)
    results.append(("set_seed runs without error", True, "seed=42"))

    try:
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            c = a @ b
        assert c.dtype == torch.bfloat16
        results.append(("bf16 autocast matmul", True, f"out dtype={c.dtype}"))
    except Exception as e:  # noqa: BLE001
        results.append(("bf16 autocast matmul", False, str(e)))

    try:
        q = torch.randn(1, 4, 32, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, 4, 32, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 32, 64, device=device, dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        results.append(("scaled_dot_product_attention on device", True, f"out shape={tuple(out.shape)}"))
    except Exception as e:  # noqa: BLE001
        results.append(("scaled_dot_product_attention on device", False, str(e)))

    try:
        import wandb  # noqa: F401

        results.append(("wandb import", True, f"version={wandb.__version__}"))
    except Exception as e:  # noqa: BLE001
        results.append(("wandb import", False, str(e)))

    mem = mem_stats()
    results.append(("mem_stats", True, ", ".join(f"{k}={v:.1f}MB" for k, v in mem.items())))

    print("\n--- Summary ---")
    all_ok = True
    for name, ok, detail in results:
        mark = CHECK if ok else CROSS
        print(f"{mark} {name:38s} {detail}")
        all_ok = all_ok and ok

    if all_ok:
        print(f"\n{CHECK} Environment OK.")
        sys.exit(0)
    else:
        print(f"\n{CROSS} Environment has problems — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
