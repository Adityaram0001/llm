"""Attention: MHA / GQA / MQA via `F.scaled_dot_product_attention`, config-selected.

`n_kv_heads` is the single knob: `== n_heads` gives ordinary multi-head attention (every query
head gets its own K/V), `== 1` gives multi-query attention (Shazeer '19, all query heads share
one K/V — much smaller KV cache), anything in between gives grouped-query attention (Ainslie
et al. '23, the LLaMA-2-70B/Mistral compromise). K/V are computed at `n_kv_heads` then
`repeat_interleave`d up to `n_heads` before the SDPA call, so the matmul shapes match; the
memory savings show up in the (smaller) K/V projection weights and in a KV cache at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norms import make_norm
from .positional import RotaryEmbedding, apply_rotary


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = make_norm(cfg.norm, cfg.head_dim)
            self.k_norm = make_norm(cfg.norm, cfg.head_dim)

        self.rotary = (
            RotaryEmbedding(cfg.head_dim, cfg.rope_theta) if cfg.pos_encoding == "rope" else None
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """`attn_bias`: precomputed ALiBi+causal bias (n_heads, T, T), or None to use fused
        causal masking (`is_causal=True`, cheaper — used for every pos_encoding except alibi)."""
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rotary is not None:
            cos, sin = self.rotary(T, x.device)
            q, k = apply_rotary(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        if attn_bias is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


def make_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attention == "mha_gqa":
        return Attention(cfg)
    if cfg.attention == "mla":
        raise NotImplementedError(
            "attention='mla' (Multi-head Latent Attention) is a phase 5-C technique — "
            "the config field exists so configs stay loadable, but it isn't implemented yet."
        )
    raise ValueError(f"Unknown attention type: {cfg.attention}")
