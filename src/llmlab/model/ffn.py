"""Feed-forward block variants: plain GELU MLP vs SwiGLU.

SwiGLU (Shazeer '20, used by LLaMA/DeepSeek/Qwen) replaces the single GELU nonlinearity with a
*gated* one: one linear projection is silu-activated and multiplies elementwise into a second,
unactivated projection, before a third projection brings it back to `d_model`. It needs 3
weight matrices instead of 2, so at matched *parameter* count its hidden dim is scaled down by
2/3 (`ffn_mult=8/3` is the conventional value that keeps SwiGLU's params ~equal to a 4x-GELU
MLP). Empirically it improves loss for the same compute in most reported ablations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELUMLP(nn.Module):
    def __init__(self, d_model: int, ffn_mult: float, dropout: float):
        super().__init__()
        hidden = int(ffn_mult * d_model)
        self.fc_in = nn.Linear(d_model, hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc_in(x))
        return self.dropout(self.fc_out(x))


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, ffn_mult: float, dropout: float):
        super().__init__()
        hidden = int(ffn_mult * d_model)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


def make_ffn(ffn_type: str, d_model: int, ffn_mult: float, dropout: float) -> nn.Module:
    if ffn_type == "gelu":
        return GELUMLP(d_model, ffn_mult, dropout)
    if ffn_type == "swiglu":
        return SwiGLUMLP(d_model, ffn_mult, dropout)
    raise ValueError(f"Unknown ffn type: {ffn_type}")
