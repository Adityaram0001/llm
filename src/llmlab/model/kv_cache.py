"""Per-layer KV caches for incremental (token-by-token) decoding — phase 5-C.

The whole point of MQA/GQA/MLA over MHA is *inference* memory: during autoregressive
generation you keep, for every past token, whatever the attention layer needs to attend to it
again. With plain MHA that is K and V at full `n_heads * head_dim` each. GQA/MQA shrink it by
sharing K/V across query heads (`n_kv_heads < n_heads`). MLA shrinks it a different way — it
caches a single compressed latent `c_kv` (`kv_lora_rank`) plus one shared decoupled-RoPE key
(`rope_head_dim`), and re-expands per-head K/V on the fly (DeepSeek-V2 §2).

`bytes_per_token()` on each cache is what Wave C measures analytically-vs-empirically: it is the
concrete "how many numbers do I store per token, per layer" that the paper's KV-cache table is
counting. Multiply by `n_layers`, dtype size, and sequence length for the full footprint.
"""

from __future__ import annotations

import torch


class KVCache:
    """Standard K/V cache for MHA/GQA/MQA. Stores post-RoPE K and V at `n_kv_heads`.

    Shapes grow along the time axis: `k`, `v` are `(B, n_kv_heads, T, head_dim)`. We cache at
    `n_kv_heads` (not the repeated `n_heads`) — that is exactly where GQA/MQA save memory.
    """

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V (`(B, n_kv_heads, t, head_dim)`) and return the full cache."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def bytes_per_token(self, dtype_size: int = 2) -> int:
        """Per-token, per-layer cache size = (K + V) numbers × dtype bytes. Independent of B/T."""
        if self.k is None:
            return 0
        n_kv_heads, head_dim = self.k.shape[1], self.k.shape[3]
        return 2 * n_kv_heads * head_dim * dtype_size

    def nbytes(self) -> int:
        if self.k is None:
            return 0
        return self.k.element_size() * (self.k.numel() + self.v.numel())


class MLACache:
    """Latent cache for MLA. Stores the compressed `c_kv` and the shared decoupled-RoPE key.

    `c_kv`: `(B, T, kv_lora_rank)` — the low-rank KV latent, re-expanded to per-head K/V on use.
    `k_rope`: `(B, 1, T, rope_head_dim)` — one RoPE key shared across all heads (decoupled).
    Per token we store `kv_lora_rank + rope_head_dim` numbers regardless of `n_heads` — that is
    MLA's cache win, and it does not grow with the number of heads the way MHA's does.
    """

    def __init__(self) -> None:
        self.c_kv: torch.Tensor | None = None
        self.k_rope: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.c_kv is None else self.c_kv.shape[1]

    def append(
        self, c_kv: torch.Tensor, k_rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `c_kv` `(B, t, kv_lora_rank)` and `k_rope` `(B, 1, t, rope_head_dim)`."""
        if self.c_kv is None:
            self.c_kv, self.k_rope = c_kv, k_rope
        else:
            self.c_kv = torch.cat([self.c_kv, c_kv], dim=1)
            self.k_rope = torch.cat([self.k_rope, k_rope], dim=2)
        return self.c_kv, self.k_rope

    def bytes_per_token(self, dtype_size: int = 2) -> int:
        if self.c_kv is None:
            return 0
        kv_lora_rank = self.c_kv.shape[2]
        rope_head_dim = self.k_rope.shape[3]
        return (kv_lora_rank + rope_head_dim) * dtype_size

    def nbytes(self) -> int:
        if self.c_kv is None:
            return 0
        return self.c_kv.element_size() * (self.c_kv.numel() + self.k_rope.numel())
