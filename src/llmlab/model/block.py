"""One transformer block: attention sub-layer + FFN sub-layer, each wrapped in a residual.

`norm_position="pre"` (GPT-2 onward, most modern models): normalize *before* the sub-layer,
so the residual stream itself is never normalized — this is what makes very deep transformers
trainable without careful warmup (Xiong et al. '20). `norm_position="post"` (original
Transformer/GPT-1): normalize *after* adding the residual — historically less stable at depth,
kept here as the ablation control.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import make_attention
from .config import ModelConfig
from .ffn import make_ffn
from .moe import MoEFFN
from .norms import make_norm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_position = cfg.norm_position
        self.attn_norm = make_norm(cfg.norm, cfg.d_model)
        self.attn = make_attention(cfg)
        self.ffn_norm = make_norm(cfg.norm, cfg.d_model)
        self.ffn = (
            MoEFFN(cfg) if cfg.moe is not None else make_ffn(cfg.ffn, cfg.d_model, cfg.ffn_mult, cfg.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        """`cache`: optional per-layer KV cache (incremental decode); None = full-sequence path."""
        if self.norm_position == "pre":
            x = x + self.attn(self.attn_norm(x), attn_bias, cache)
            x = x + self.ffn(self.ffn_norm(x))
        else:
            x = self.attn_norm(x + self.attn(x, attn_bias, cache))
            x = self.ffn_norm(x + self.ffn(x))
        return x
