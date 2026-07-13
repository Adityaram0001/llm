"""Attention variants — phase 3 MHA/GQA/MQA + phase 5-C MLA, all config-selected.

`Attention` (`attention="mha_gqa"`): `n_kv_heads` is the single knob. `== n_heads` gives ordinary
multi-head attention (every query head its own K/V); `== 1` gives multi-query attention
(Shazeer '19, all query heads share one K/V — tiny KV cache); in between gives grouped-query
attention (Ainslie et al. '23, the LLaMA-2-70B/Mistral compromise). K/V are computed at
`n_kv_heads` then `repeat_interleave`d up to `n_heads` for the matmul; the savings live in the
smaller K/V projection weights and — the point of Wave C — a smaller KV cache at inference.

`MLAAttention` (`attention="mla"`): Multi-head Latent Attention (DeepSeek-V2 §2). Instead of
caching per-head K/V, it caches one low-rank latent `c_kv` (`kv_lora_rank`) plus one shared
decoupled-RoPE key (`rope_head_dim`), re-expanding per-head K/V on the fly. See
`docs/../notebooks/06_mla_explained.ipynb` for the matrix diagrams.

Both modules accept an optional per-layer `cache` (see `kv_cache.py`) for token-by-token
decoding; `cache=None` is the ordinary full-sequence training/eval path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import KVCache, MLACache
from .norms import make_norm
from .positional import RotaryEmbedding, apply_rotary


def _causal_bool_mask(
    q_len: int, past_len: int, device: torch.device
) -> torch.Tensor:
    """Boolean SDPA mask (True = attend) for queries at positions `past_len..past_len+q_len-1`
    against keys at `0..past_len+q_len-1`. Query i may attend key j iff j <= past_len+i."""
    total = past_len + q_len
    q_pos = torch.arange(past_len, total, device=device)[:, None]
    k_pos = torch.arange(total, device=device)[None, :]
    return q_pos >= k_pos


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

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """`attn_bias`: precomputed ALiBi+causal bias (n_heads, T, T), or None for fused causal
        masking (cheaper — used for every pos_encoding except alibi). `cache`: if given, append
        this step's K/V and attend against the full cached history (incremental decode); ALiBi is
        not supported together with a cache (Wave C decodes are all RoPE-based)."""
        B, T, _ = x.shape
        past_len = cache.seq_len if cache is not None else 0

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rotary is not None:
            cos, sin = self.rotary(T, x.device, offset=past_len)
            q, k = apply_rotary(q, k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)  # k, v now span the full past+current window

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        if cache is not None:
            mask = _causal_bool_mask(T, past_len, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        elif attn_bias is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2 §2).

    Query path (low-rank, params only — queries aren't cached):
        c_q   = RMSNorm(W_DQ · h)                        (B,T,q_lora_rank)
        q     = W_UQ · c_q -> per head [q_nope | q_rope] (nope_head_dim + rope_head_dim)
    Key/Value path (low-rank, the *latent* is what gets cached):
        c_kv  = W_DKV · h                                (B,T,kv_lora_rank)   <- cached
        [k_nope | v] = W_UKV · RMSNorm(c_kv)  per head   (nope_head_dim + v_head_dim)
    Decoupled RoPE key (one head, shared, carries all the positional signal):
        k_rope = RoPE(W_KR · h)                          (B,1,T,rope_head_dim) <- cached
    Per head: q=[q_nope|q_rope], k=[k_nope|k_rope(shared)], attend, value=v, then W_O.

    Only `q_rope`/`k_rope` see RoPE; `q_nope`/`k_nope` are position-free. That decoupling is what
    lets the cached latent stay position-agnostic (re-expandable at any offset) while still
    encoding position — the paper's key trick.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.mla is not None
        m = cfg.mla
        self.n_heads = cfg.n_heads
        self.nope = m.nope_head_dim
        self.rope = m.rope_head_dim
        self.v_head_dim = m.v_head_dim
        self.kv_lora_rank = m.kv_lora_rank
        self.q_lora_rank = m.q_lora_rank
        self.dropout = cfg.dropout
        self.qk_head_dim = self.nope + self.rope

        # query: down-project, norm, up-project into per-head [nope | rope]
        self.q_down = nn.Linear(cfg.d_model, m.q_lora_rank, bias=False)
        self.q_norm = make_norm(cfg.norm, m.q_lora_rank)
        self.q_up = nn.Linear(m.q_lora_rank, cfg.n_heads * self.qk_head_dim, bias=False)

        # key/value: down-project to the cached latent, norm, up-project into per-head [nope | v]
        self.kv_down = nn.Linear(cfg.d_model, m.kv_lora_rank, bias=False)
        self.kv_norm = make_norm(cfg.norm, m.kv_lora_rank)
        self.kv_up = nn.Linear(m.kv_lora_rank, cfg.n_heads * (self.nope + self.v_head_dim), bias=False)

        # decoupled RoPE key: single shared head, straight from h (bypasses the latent)
        self.k_rope_proj = nn.Linear(cfg.d_model, self.rope, bias=False)

        self.o_proj = nn.Linear(cfg.n_heads * self.v_head_dim, cfg.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.rope, cfg.rope_theta)

    def _expand_kv(
        self, c_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Up-project the cached latent `c_kv` (B,S,kv_lora_rank) into per-head content keys
        `k_nope` (B,H,S,nope) and values `v` (B,H,S,v_head_dim)."""
        B, S, _ = c_kv.shape
        kv = self.kv_up(self.kv_norm(c_kv)).view(B, S, self.n_heads, self.nope + self.v_head_dim)
        kv = kv.transpose(1, 2)  # (B,H,S,nope+v)
        k_nope, v = kv.split([self.nope, self.v_head_dim], dim=-1)
        return k_nope, v

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,  # unused: MLA owns its (decoupled RoPE) positions
        cache: MLACache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        past_len = cache.seq_len if cache is not None else 0

        # --- query ---
        c_q = self.q_norm(self.q_down(x))
        q = self.q_up(c_q).view(B, T, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)  # (B,H,T,nope),(B,H,T,rope)

        # --- kv latent (cached) + decoupled rope key (cached) ---
        c_kv = self.kv_down(x)  # (B,T,kv_lora_rank)
        k_rope = self.k_rope_proj(x).view(B, T, 1, self.rope).transpose(1, 2)  # (B,1,T,rope)

        # RoPE on the current window (offset by cache length for correct absolute positions)
        cos, sin = self.rotary(T, x.device, offset=past_len)
        q_rope, k_rope = apply_rotary(q_rope, k_rope, cos, sin)

        if cache is not None:
            c_kv, k_rope = cache.append(c_kv, k_rope)  # now span full past+current

        k_nope, v = self._expand_kv(c_kv)  # (B,H,S,nope),(B,H,S,v)
        S = k_nope.shape[2]
        k_rope = k_rope.expand(B, self.n_heads, S, self.rope)  # share the decoupled key per head

        q_full = torch.cat([q_nope, q_rope], dim=-1)  # (B,H,T,nope+rope)
        k_full = torch.cat([k_nope, k_rope], dim=-1)  # (B,H,S,nope+rope)

        dropout_p = self.dropout if self.training else 0.0
        if cache is not None:
            mask = _causal_bool_mask(T, past_len, x.device)
            out = F.scaled_dot_product_attention(q_full, k_full, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            out = F.scaled_dot_product_attention(q_full, k_full, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_head_dim)
        return self.o_proj(out)


def make_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attention == "mha_gqa":
        return Attention(cfg)
    if cfg.attention == "mla":
        return MLAAttention(cfg)
    raise ValueError(f"Unknown attention type: {cfg.attention}")


def make_cache(cfg: ModelConfig) -> KVCache | MLACache:
    """One fresh per-layer cache matching the model's attention type."""
    return MLACache() if cfg.attention == "mla" else KVCache()
