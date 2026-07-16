"""ModelConfig: every architecture technique this lab studies is a field here, not a new file.

Baseline path (phase 3) implements: mha_gqa attention, learned/sinusoidal/rope/alibi/none
positions, layernorm/rmsnorm, gelu/swiglu FFN, weight tying. `moe`/`mtp`/`attention="mla"` are
config fields that exist now (so configs stay loadable across phases) but raise
NotImplementedError until phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import yaml


@dataclass
class MLAConfig:
    """Multi-head Latent Attention (DeepSeek-V2 §2) — phase 5-C.

    The KV cache stores only the compressed latent `c_kv` (`kv_lora_rank`) plus one shared
    decoupled-RoPE key (`rope_head_dim`) per token — NOT per-head K and V. Per head, query/key
    each split into a content part (`nope_head_dim`, position-free) and a decoupled RoPE part
    (`rope_head_dim`); the value head is `v_head_dim`. Query is itself low-rank-compressed to
    `q_lora_rank` (params only — queries aren't cached). Head-dim-preserving S-tier default:
    nope=32, rope=32 → per-head Q/K dim 64 (== baseline head_dim), v=64.
    """

    kv_lora_rank: int
    q_lora_rank: int
    rope_head_dim: int
    nope_head_dim: int
    v_head_dim: int


@dataclass
class MoEConfig:
    """Mixture-of-Experts routing — phase 5-F."""

    n_experts: int
    n_shared: int
    top_k: int
    balancing: Literal["aux_loss", "bias_free"] = "aux_loss"
    aux_loss_weight: float = 0.01  # coefficient on the Switch-style balancing loss (balancing="aux_loss")
    bias_update_rate: float = 0.001  # per-step bias nudge (DeepSeek-V3, balancing="bias_free")


@dataclass
class MTPConfig:
    """Multi-Token Prediction (DeepSeek-V3) — phase 5-F."""

    n_predict_tokens: int
    loss_weight: float = 0.3  # lambda on the averaged MTP cross-entropy (DeepSeek-V3's own default)


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int  # == n_heads -> MHA, 1 -> MQA, else GQA
    head_dim: int
    max_seq_len: int
    dropout: float = 0.0

    norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    norm_position: Literal["pre", "post"] = "pre"
    qk_norm: bool = False

    pos_encoding: Literal["learned", "sinusoidal", "rope", "alibi", "none"] = "rope"
    rope_theta: float = 10000.0

    ffn: Literal["gelu", "swiglu"] = "swiglu"
    ffn_mult: float = 8 / 3

    attention: Literal["mha_gqa", "mla"] = "mha_gqa"
    mla: Optional[MLAConfig] = None

    tie_embeddings: bool = True
    init: Literal["gpt2", "scaled"] = "gpt2"

    moe: Optional[MoEConfig] = None
    mtp: Optional[MTPConfig] = None

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.attention == "mla" and self.mla is None:
            raise ValueError("attention='mla' requires an `mla:` config block")
        if self.moe is not None and not isinstance(self.moe, MoEConfig):
            self.moe = MoEConfig(**self.moe)
        if self.mtp is not None and not isinstance(self.mtp, MTPConfig):
            self.mtp = MTPConfig(**self.mtp)
        if self.mla is not None and not isinstance(self.mla, MLAConfig):
            self.mla = MLAConfig(**self.mla)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
