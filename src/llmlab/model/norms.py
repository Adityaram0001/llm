"""LayerNorm and RMSNorm, plus a factory keyed off ModelConfig.norm.

RMSNorm (Zhang & Sennrich '19) drops LayerNorm's mean-centering and learned bias — it turns out
re-centering activations barely matters for transformers, so RMSNorm gets ~the same effect
(rescale by the RMS of the activations, so gradients don't explode/vanish through depth) for
less compute. Used by LLaMA, DeepSeek, Qwen and most modern open models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 for stability, then cast back (matters under bf16 autocast).
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm type: {norm_type}")
