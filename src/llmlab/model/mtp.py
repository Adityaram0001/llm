"""Multi-Token Prediction (Gloeckle et al. '24; DeepSeek-V3 S2.2) -- phase 5-F, flagship 3.

Standard next-token training only ever supervises "hidden state at i -> predict token i+1". MTP
adds `n_predict_tokens` extra *sequential* heads: depth d combines the previous depth's hidden
state at position i with the TRUE (teacher-forced) embedding of token i+d, runs one more
transformer block over the (now d shorter) sequence, and predicts token i+d+1 through the SAME
shared final_norm+lm_head the main model uses. The claim: denser training signal per token
improves sample efficiency, and the causal chain of MTP modules is a cheap proxy for training
representations that are useful a few tokens ahead, not just one.

Deliberately simplified vs. the full V3 design: the MTP block here is always a DENSE (non-MoE)
`Block` even when the main trunk uses MoE FFN layers -- avoids nesting a second, independent
MoE router+expert set behind one extra head at S-tier scale, where the interesting comparison is
"does MTP help loss at fixed compute", not MTP-of-MoE.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from .block import Block
from .config import ModelConfig
from .norms import make_norm


class MTPHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dense_cfg = replace(cfg, moe=None, mtp=None)
        self.norm_h = make_norm(cfg.norm, cfg.d_model)
        self.norm_emb = make_norm(cfg.norm, cfg.d_model)
        self.combine = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(dense_cfg)

    def forward(
        self,
        h_prev: torch.Tensor,
        next_tok_emb: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        combined = self.combine(
            torch.cat([self.norm_h(h_prev), self.norm_emb(next_tok_emb)], dim=-1)
        )
        return self.block(combined, attn_bias)
