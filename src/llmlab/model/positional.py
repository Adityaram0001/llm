"""Positional encoding variants — a config switch, not five model files.

Two different mechanisms live here:
- **Additive to the input embedding** (learned, sinusoidal): produces a `(seq_len, d_model)`
  tensor added once, before the first block. The model has to *learn* to recover relative
  position from these absolute vectors.
- **Injected inside attention** (RoPE, ALiBi): every block re-derives position information
  from Q/K directly, which is why these generalize better to sequences longer than anything
  seen in training (RoPE: rotate; ALiBi: additive distance penalty on attention logits).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """A plain lookup table, one learned vector per absolute position (GPT-1/GPT-2 style)."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.emb(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed (non-learned) sin/cos table (Vaswani et al. '17). No parameters to train."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:seq_len].to(device)


class RotaryEmbedding(nn.Module):
    """RoPE (Su et al. '21): rotate each (q, k) pair by an angle proportional to its position.

    Encodes *relative* position implicitly — the dot product of two rotated vectors depends
    only on their position difference, not their absolute positions (see `rotary_relative_shift
    property` test). No added parameters; `theta` controls how quickly the rotation angle
    grows across frequency bands (higher theta -> better long-context extrapolation).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq.to(device))  # (seq_len, head_dim/2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q, k of shape (batch, n_heads, seq_len, head_dim); cos/sin: (seq_len, head_dim)."""
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Per-head linear-bias slopes (Press et al. '21). Geometric sequence, closed form for
    powers of two; falls back to interpolation otherwise (see the paper's appendix)."""

    def slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(slopes_power_of_2(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    base = slopes_power_of_2(closest)
    extra = slopes_power_of_2(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(base + extra)


def build_alibi_bias(n_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Additive attention bias combining ALiBi's distance penalty with the causal mask.

    Shape (n_heads, seq_len, seq_len), to be passed as `attn_mask` to
    `F.scaled_dot_product_attention` (with `is_causal=False`).
    """
    slopes = alibi_slopes(n_heads).to(device)
    positions = torch.arange(seq_len, device=device)
    distance = positions[None, :] - positions[:, None]  # (seq_len, seq_len), key - query
    bias = slopes[:, None, None] * distance[None, :, :].float()  # (n_heads, seq_len, seq_len)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    return bias + causal_mask[None, :, :]
