"""Environment utilities: seeding, device selection, param counting, memory stats."""

from __future__ import annotations

import os
import random

import numpy as np
import psutil
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA + MPS) RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Return the best available device: cuda (cloud GPU) > mps (this Mac) > cpu.

    All project code must go through this — never hard-code "mps" or "cuda" —
    so the same scripts run locally and on rented Linux/NVIDIA boxes (docs/CLOUD.md).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Mixed-precision context for the given device (bf16 on both MPS and CUDA).

    On CPU returns bf16 autocast too (slow but numerically consistent).
    Usage: `with autocast_ctx(device): logits, loss = model(x, y)`
    """
    return torch.autocast(device_type=device.type, dtype=dtype)


def param_count(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters (trainable by default)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def mem_stats() -> dict[str, float]:
    """Return current process RSS and accelerator-allocated memory, in MB."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    stats = {"rss_mb": rss_mb}
    if torch.cuda.is_available():
        stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
    if torch.backends.mps.is_available():
        stats["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024**2)
        stats["mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024**2)
    return stats
