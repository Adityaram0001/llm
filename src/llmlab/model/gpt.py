"""GPT: embeddings -> transformer blocks -> final norm -> LM head.

Every technique this lab studies (norm type/position, positional encoding, attention variant,
FFN type, weight tying, init scheme, MoE routing, MTP heads) is a `ModelConfig` field threaded
through here — there is exactly one model class, not one per technique.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .block import Block
from .config import ModelConfig
from .ffn import GELUMLP, SwiGLUMLP
from .moe import MoEFFN
from .mtp import MTPHead
from .norms import make_norm
from .positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    build_alibi_bias,
)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.mtp is not None and cfg.pos_encoding in ("learned", "sinusoidal"):
            # MTP heads run on a combined [prev hidden; next-token embedding] tensor that never
            # passes through the input embedding stage, which is the only place learned/
            # sinusoidal positions get added — rope/alibi/none all inject position inside
            # Attention itself (per-block), so they work unmodified on the shorter MTP
            # subsequences (see `_mtp_loss`).
            raise NotImplementedError(
                "MTP requires pos_encoding in {rope, alibi, none} — learned/sinusoidal positions "
                "are added at the input embedding stage, which MTP heads never see."
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

        self.mtp_heads = (
            nn.ModuleList([MTPHead(cfg) for _ in range(cfg.mtp.n_predict_tokens)])
            if cfg.mtp is not None
            else None
        )

        self.apply(self._init_weights)
        self._scale_residual_projections()

        # Wave E (phase 5): gradient checkpointing is a runtime memory/compute trade-off, not
        # an architecture choice, so it's a plain attribute (set by Trainer from TrainConfig)
        # rather than a ModelConfig field -- toggling it doesn't change what the model computes,
        # only whether each block's activations are recomputed on the backward pass instead of
        # kept in memory (Chen et al. '16). Only applies during training with no KV cache.
        self.gradient_checkpointing = False

        # Wave F (phase 5): populated by forward() whenever targets is given -- moe_aux_loss,
        # expert_load, mtp_loss, read by Trainer for metrics.jsonl/wandb logging.
        self.last_aux_metrics: dict = {}

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
            residual_writing = [b.attn.o_proj for b in self.blocks]
            for b in self.blocks:
                residual_writing.extend(self._ffn_residual_projections(b.ffn))
            if self.mtp_heads is not None:
                for head in self.mtp_heads:
                    residual_writing.append(head.block.attn.o_proj)
                    residual_writing.extend(self._ffn_residual_projections(head.block.ffn))
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

    @staticmethod
    def _ffn_residual_projections(ffn: nn.Module) -> list[nn.Linear]:
        """The linear layer(s) of an FFN sub-layer that write directly into the residual
        stream -- one per dense FFN, or one per expert (routed + shared) for `MoEFFN`, since
        every active expert writes into the same residual stream on the tokens it's assigned."""
        if isinstance(ffn, GELUMLP):
            return [ffn.fc_out]
        if isinstance(ffn, MoEFFN):
            return [e.down_proj for e in ffn.experts] + [e.down_proj for e in ffn.shared_experts]
        return [ffn.down_proj]  # SwiGLUMLP

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`caches`: optional list of per-layer KV caches for incremental decode (one per block).
        Cached decode is only wired for on-the-fly position encodings (rope / alibi-free / MLA);
        with a cache the token positions are offset by the cache length."""
        B, T = idx.shape
        past_len = caches[0].seq_len if caches is not None else 0
        if (
            T + past_len > self.cfg.max_seq_len
            and self.cfg.pos_encoding in ("learned", "sinusoidal")
        ):
            # Only these two are physically bounded by max_seq_len (fixed-size lookup
            # table/precomputed pe table). RoPE/ALiBi/none/MLA derive position info on the fly per
            # forward call, so they can run at any T -- that's exactly what makes the phase-5
            # Wave B length-extrapolation probe (train@512, eval ppl@1024/2048) possible (RW-5).
            raise ValueError(f"sequence length {T + past_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        if caches is not None and self.pos_emb is not None:
            raise ValueError("cached decode is not supported for learned/sinusoidal encodings")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(T, idx.device)
        x = self.drop(x)

        attn_bias = None
        if self.cfg.pos_encoding == "alibi":
            if caches is not None:
                raise ValueError("cached decode is not supported for ALiBi")
            attn_bias = build_alibi_bias(self.cfg.n_heads, T, idx.device)

        use_ckpt = self.gradient_checkpointing and self.training and caches is None
        for i, block in enumerate(self.blocks):
            cache = None if caches is None else caches[i]
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(block, x, attn_bias, cache, use_reentrant=False)
            else:
                x = block(x, attn_bias, cache)
        trunk_h = x  # pre-final-norm hidden -- MTP heads branch off this, not off `logits`
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        self.last_aux_metrics = {}
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            # Pure next-token CE, before any aux term is mixed in -- Trainer.evaluate() reads
            # THIS for val_loss so Wave F stays comparable to every other wave's noise-floor
            # convention (docs/EXPERIMENTS.md). `loss` below keeps accumulating aux terms since
            # that combined value is what train_step() actually backprops through.
            self.last_aux_metrics["ce_loss"] = float(loss.detach())

            if self.cfg.moe is not None:
                moe_aux_loss = sum(block.ffn.last_aux_loss for block in self.blocks)
                loss = loss + self.cfg.moe.aux_loss_weight * moe_aux_loss
                self.last_aux_metrics["moe_aux_loss"] = float(moe_aux_loss.detach())
                self.last_aux_metrics["expert_load"] = [
                    block.ffn.last_expert_load.tolist() for block in self.blocks
                ]

            if self.mtp_heads is not None:
                mtp_loss = self._mtp_loss(trunk_h, targets)
                if mtp_loss is not None:
                    loss = loss + self.cfg.mtp.loss_weight * mtp_loss
                    self.last_aux_metrics["mtp_loss"] = float(mtp_loss.detach())

        return logits, loss

    def _mtp_loss(self, trunk_h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor | None:
        """Chains the MTP heads sequentially -- depth d combines depth (d-1)'s hidden state
        (dropping its last position) with the embedding of the token d steps ahead (teacher-
        forced from `targets`, itself already the next-token-shifted sequence: `targets[:,j]`
        IS the token at absolute position j+1), then predicts the token d+1 steps ahead.

        Sequence-length bookkeeping: depth d's inputs/targets both have length T-d (T = trunk_h's
        length); every depth's retained positions are the ORIGINAL sequence's first T-d
        positions (we always trim from the right), so RoPE/ALiBi need no offset — the subsequence
        is exactly the causal prefix the position encoding already expects.
        """
        h = trunk_h
        y = targets
        T = y.shape[1]
        losses = []
        for depth, head in enumerate(self.mtp_heads, start=1):
            if h.shape[1] <= 1:
                break
            h_prev = h[:, :-1, :]
            next_emb = self.tok_emb(y[:, depth - 1 : T - 1])
            bias = None
            if self.cfg.pos_encoding == "alibi":
                bias = build_alibi_bias(self.cfg.n_heads, h_prev.shape[1], h_prev.device)
            h = head(h_prev, next_emb, bias)
            target_d = y[:, depth:]
            depth_logits = self.lm_head(self.final_norm(h))
            losses.append(
                F.cross_entropy(depth_logits.reshape(-1, depth_logits.size(-1)), target_d.reshape(-1))
            )
        return torch.stack(losses).mean() if losses else None

    def update_moe_bias(self, update_rate: float) -> None:
        """Called by `Trainer` once per optimizer step (after all grad-accum micro-batches) --
        no-op unless `moe.balancing == "bias_free"`; see `MoEFFN.update_bias`."""
        if self.cfg.moe is None:
            return
        for block in self.blocks:
            block.ffn.update_bias(update_rate)

    # -- generation ------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling. `top_k`: keep only the k highest-probability tokens.
        `top_p` (nucleus, Holtzman et al. '19): keep the smallest set of tokens whose
        cumulative probability >= top_p. Both, if set, apply on top of temperature scaling.

        `use_cache=True` prefills the prompt once then feeds ONE new token per step against a
        per-layer KV cache (the O(1)-per-step decode that MQA/GQA/MLA optimize the memory of);
        `use_cache=False` falls back to re-running the whole prefix each step (correct but O(T)),
        used mainly by learned/sinusoidal/alibi configs the cache path doesn't cover."""
        from .attention import make_cache

        was_training = self.training
        self.eval()
        caches = None
        if use_cache and self.pos_emb is None and self.cfg.pos_encoding != "alibi":
            caches = [make_cache(self.cfg) for _ in range(self.cfg.n_layers)]
        for step in range(max_new_tokens):
            if caches is not None:
                # prefill the whole prompt on step 0, then only the last token each step
                idx_cond = idx if step == 0 else idx[:, -1:]
                logits, _ = self(idx_cond, caches=caches)
            else:
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

        counts = {
            "embed": self.tok_emb.weight.numel(),
            "pos_embed": 0,
            "attn": 0,
            "ffn": 0,
            "norms": 0,
            "head": 0,
            "mtp": 0,
        }
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
        if self.mtp_heads is not None:
            counts["mtp"] = sum(p.numel() for head in self.mtp_heads for p in head.parameters())
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
