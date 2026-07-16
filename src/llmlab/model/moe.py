"""DeepSeekMoE routing (DeepSeek-MoE '24) + DeepSeek-V3's aux-loss-free load balancing --
phase 5-F, flagship 3.

Fine-grained experts: instead of a few large FFN experts, DeepSeekMoE uses many SMALL experts
(here: `n_experts` routed + `n_shared` always-on) so the top-k combination can mix finer-grained
knowledge per token. Each expert's hidden dim is shrunk so the ACTIVE params per token
(n_shared + top_k expert-equivalents) match the dense baseline's single FFN -- more total
capacity, identical active compute, which is the paper's headline claim.

Two balancing methods, config-selected (`MoEConfig.balancing`):
- "aux_loss" (Shazeer/Switch/GShard-style): an auxiliary loss term added to the main LM loss
  that penalizes correlation between routing probability and token count per expert -- a soft,
  gradient-based nudge toward balance that competes with the main objective.
- "bias_free" (DeepSeek-V3 S2.1.2): a per-expert bias added to the routing logits BEFORE top-k
  SELECTION only (never to the combination weight, never part of any loss) -- updated by a
  tiny sign-based rule after every optimizer step. Zero gradient interference with the main
  objective, which is the whole point of the method.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .ffn import SwiGLUMLP


class MoEFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.moe is not None
        m = cfg.moe
        self.n_experts = m.n_experts
        self.top_k = m.top_k
        self.balancing = m.balancing

        # Active-param match: n_shared+top_k expert-equivalents run per token, so each expert's
        # hidden dim is the dense FFN's hidden dim divided by that count (fine-grained
        # segmentation, DeepSeekMoE section 3.1).
        dense_hidden = int(cfg.ffn_mult * cfg.d_model)
        expert_hidden = max(1, round(dense_hidden / (m.n_shared + m.top_k)))
        expert_mult = expert_hidden / cfg.d_model

        self.router = nn.Linear(cfg.d_model, m.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUMLP(cfg.d_model, expert_mult, cfg.dropout) for _ in range(m.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUMLP(cfg.d_model, expert_mult, cfg.dropout) for _ in range(m.n_shared)]
        )

        if self.balancing == "bias_free":
            self.register_buffer("routing_bias", torch.zeros(m.n_experts))
            self.register_buffer("_load_accum", torch.zeros(m.n_experts))

        # Set fresh every forward call; read by GPT.forward (loss) and Trainer (logging).
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_expert_load: torch.Tensor = torch.zeros(m.n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        logits = self.router(x_flat)  # (N, n_experts)
        gate_probs = F.softmax(logits, dim=-1)  # differentiable combine weights, bias-free

        select_logits = logits
        if self.balancing == "bias_free":
            select_logits = logits + self.routing_bias  # bias shifts SELECTION only
        _, topk_idx = select_logits.topk(self.top_k, dim=-1)  # (N, top_k)
        gate_sel = gate_probs.gather(-1, topk_idx)
        gate_sel = gate_sel / gate_sel.sum(dim=-1, keepdim=True)  # renormalize top-k to sum 1

        out = torch.zeros_like(x_flat)
        load_counts = x_flat.new_zeros(self.n_experts)
        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = (topk_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = gate_sel[token_idx, slot_idx].unsqueeze(-1)
            # under bf16 autocast, `expert(...)`'s Linear ops autocast their output to bf16
            # regardless of `out`'s (fp32 residual-stream) dtype -- index_add_ has no implicit
            # type promotion the way `+` does, so it needs an explicit cast.
            contribution = (weight * expert(x_flat[token_idx])).to(out.dtype)
            out.index_add_(0, token_idx, contribution)
            load_counts[e] = token_idx.numel()

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        self.last_expert_load = (load_counts / N).detach()
        if self.training and self.balancing == "bias_free":
            self._load_accum += load_counts.detach()

        if self.balancing == "aux_loss":
            f = load_counts.detach() / (N * self.top_k)  # routed-slot fraction (stop-grad)
            p = gate_probs.mean(dim=0)  # differentiable average routing probability mass
            self.last_aux_loss = self.n_experts * (f * p).sum()
        else:
            self.last_aux_loss = torch.zeros((), device=x.device)

        return out.view(B, T, D)

    def update_bias(self, update_rate: float) -> None:
        """DeepSeek-V3 S2.1.2: a fixed-size, gradient-free nudge per expert -- no aux loss term,
        no interference with the main objective. Overloaded experts (more tokens than the
        per-expert average this step) get their bias nudged down; underloaded experts get
        nudged up, making them relatively more likely to be selected next step. Called once per
        optimizer step by `Trainer`, aggregating load across all of that step's grad-accum
        micro-batches."""
        if self.balancing != "bias_free":
            return
        avg = self._load_accum.mean()
        self.routing_bias += update_rate * torch.sign(avg - self._load_accum)
        self._load_accum.zero_()
