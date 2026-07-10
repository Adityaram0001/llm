"""Environment utilities: seeding, device selection, param counting, memory stats."""

from __future__ import annotations

import os
import random

import numpy as np
import psutil
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + MPS) RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Return the MPS device if available, else fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_count(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters (trainable by default)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def mem_stats() -> dict[str, float]:
    """Return current process RSS and MPS-allocated memory, in MB."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    stats = {"rss_mb": rss_mb}
    if torch.backends.mps.is_available():
        stats["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024**2)
        stats["mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024**2)
    return stats
