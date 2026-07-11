"""GPT: embeddings -> transformer blocks -> final norm -> LM head.

Every technique this lab studies (norm type/position, positional encoding, attention variant,
FFN type, weight tying, init scheme) is a `ModelConfig` field threaded through here — there is
exactly one model class, not one per technique. `moe`/`mtp` fields exist so old configs stay
loadable but raise `NotImplementedError` until phase 5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .config import ModelConfig
from .ffn import GELUMLP, SwiGLUMLP
from .norms import make_norm
from .positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    build_alibi_bias,
)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.moe is not None:
            raise NotImplementedError(
                "MoE (Mixture-of-Experts) is a phase 5-F technique — the config field exists "
                "so configs stay loadable, but routing isn't implemented yet."
            )
        if cfg.mtp is not None:
            raise NotImplementedError(
                "MTP (Multi-Token Prediction) is a phase 5-F technique — the config field "
                "exists so configs stay loadable, but it isn't implemented yet."
            )
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb: nn.Module | None
        if cfg.pos_encoding == "learned":
            self.pos_emb = LearnedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        else:  # rope / alibi / none — no additive embedding at the input
            self.pos_emb = None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.norm, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # -- init --------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 init (Radford et al. '19): weights ~ N(0, 0.02), biases 0, LayerNorm/RMSNorm
        weights already default to 1 (untouched here). 0.02 is an empirical GPT-2 constant —
        small enough that logits stay near-uniform at init (loss ~ ln(vocab), see
        test_model.py) regardless of d_model, since it's not scaled by fan-in."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """Scale the output projection of every sub-layer that writes into the residual stream
        by 1/sqrt(2*n_layers) (GPT-2 §2.3 / Radford et al. '19).

        Each block adds TWO things to the residual stream (attention out, FFN out); with `n`
        blocks that's `2n` additions of roughly independent noise, so the residual stream's
        variance would grow ~linearly with depth without this — scaling each contribution down
        by 1/sqrt(2n) keeps the sum's variance ~constant regardless of depth, which is what
        lets deep transformers train without extra warmup tricks.

        `init="scaled"` applies the same 1/sqrt(2n) factor to EVERY linear layer instead of
        only the residual-writing ones (a stricter, uniformly-scaled variant seen in some
        GPT-NeoX-style configs) — worth comparing against GPT-2 init as a P5 ablation.
        """
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        if self.cfg.init == "gpt2":
            residual_writing = [
                *[b.attn.o_proj for b in self.blocks],
                *[
                    b.ffn.fc_out if isinstance(b.ffn, GELUMLP) else b.ffn.down_proj
                    for b in self.blocks
                ],
            ]
            for module in residual_writing:
                with torch.no_grad():
                    module.weight.mul_(scale)
        elif self.cfg.init == "scaled":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    with torch.no_grad():
                        module.weight.mul_(scale)
        else:
            raise ValueError(f"Unknown init scheme: {self.cfg.init}")

    # -- forward -------------------------------------------------------------

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(T, idx.device)
        x = self.drop(x)

        attn_bias = None
        if self.cfg.pos_encoding == "alibi":
            attn_bias = build_alibi_bias(self.cfg.n_heads, T, idx.device)

        for block in self.blocks:
            x = block(x, attn_bias)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    # -- generation ------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. `top_k`: keep only the k highest-probability tokens.
        `top_p` (nucleus, Holtzman et al. '19): keep the smallest set of tokens whose
        cumulative probability >= top_p. Both, if set, apply on top of temperature scaling."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                remove = cum_probs - probs > top_p  # keep the first token that crosses top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter_(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx

    # -- introspection -------------------------------------------------------

    def num_params(self, breakdown: bool = False) -> int | dict[str, int]:
        if not breakdown:
            return sum(p.numel() for p in self.parameters())

        counts = {"embed": self.tok_emb.weight.numel(), "pos_embed": 0, "attn": 0, "ffn": 0, "norms": 0, "head": 0}
        if self.pos_emb is not None:
            counts["pos_embed"] = sum(p.numel() for p in self.pos_emb.parameters())
        for block in self.blocks:
            counts["attn"] += sum(p.numel() for p in block.attn.parameters())
            counts["ffn"] += sum(p.numel() for p in block.ffn.parameters())
            counts["norms"] += sum(p.numel() for p in block.attn_norm.parameters())
            counts["norms"] += sum(p.numel() for p in block.ffn_norm.parameters())
        counts["norms"] += sum(p.numel() for p in self.final_norm.parameters())
        if not self.cfg.tie_embeddings:
            counts["head"] = self.lm_head.weight.numel()
        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        return counts

    def estimate_flops_per_token(
        self, seq_len: int | None = None, include_attn_term: bool = True
    ) -> float:
        """Kaplan et al. '20 approximation: ~6 FLOPs per (compute-contributing) parameter per
        token for a fwd+bwd pass. Excludes the token/positional embedding *lookups* (no matmul)
        but includes the final unembedding matmul even when its weight is tied to the input
        embedding, since the compute happens regardless of weight sharing. Optionally adds the
        attention mechanism's own quadratic-in-context-length term (12 * n_layers * d_model *
        seq_len), which the 6N shortcut ignores and which matters once seq_len ~ d_model."""
        seq_len = seq_len or self.cfg.max_seq_len
        b = self.num_params(breakdown=True)
        compute_params = b["attn"] + b["ffn"] + b["norms"] + self.cfg.vocab_size * self.cfg.d_model
        flops = 6 * compute_params
        if include_attn_term:
            flops += 12 * self.cfg.n_layers * self.cfg.d_model * seq_len
        return flops
