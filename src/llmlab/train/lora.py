"""LoRA from scratch (Hu et al. '21) — phase 8, Part B.

Low-Rank Adaptation freezes the pretrained weight `W` and learns a low-rank *update* alongside it:

    h = W x  +  (alpha / r) · B A x           A: (r, in)  B: (out, r),  r << min(in, out)

Only `A` and `B` train; `W` is frozen. Two matrices of rank `r` replace a full `(out, in)` update,
so the trainable-parameter count drops from `out·in` to `r·(out + in)` per adapted layer — for our
S-tier attention projection (192×192) at r=8 that's 3072 vs 36864, ~12x fewer.

**Why B is initialized to zero (and A random).** At step 0 the adapter output is
`(alpha/r)·B·A·x = 0` because `B = 0`, so the adapted model is *bit-identical* to the pretrained
model — fine-tuning starts from the base's behavior, not a random perturbation of it. Gradients
still flow: `dL/dB ∝ (dL/dh)·(A x)ᵀ` is nonzero because `A` is random, so `B` moves on step 1;
once `B ≠ 0`, `A` moves too. (If *both* were zero the adapter would be dead — `dL/dA ∝ Bᵀ(dL/dh)`
would also be zero — so exactly one of the pair must be nonzero at init.)

**Where the memory win actually comes from.** Not the frozen forward pass (that's the same FLOPs
and activations as full FT). It's the optimizer: AdamW keeps a gradient + two fp32 moments per
*trainable* parameter (~12 bytes each). Freezing 99% of the model removes 99% of that state. At
10M params the absolute saving is small (a few MB); at 7B+ it's the difference between fitting on
one GPU or not — see the exact numbers `scripts/sft.py --config …_lora_… ` logs and the Part-B
table in `docs/results/finetune_report.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Target presets: which leaf `nn.Linear` names to adapt (dense S/M models — MLA/MoE have others).
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_TARGETS = ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out")  # swiglu + gelu names
TARGET_PRESETS = {
    "attn": ATTN_TARGETS,
    "attn+ffn": ATTN_TARGETS + FFN_TARGETS,
    "ffn": FFN_TARGETS,
}


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a rank-`r` additive update `W x + (alpha/r)·B A x`."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A: (r, in), B: (out, r). A ~ kaiming-uniform (nonzero); B = 0 so the adapter starts at 0.
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Use F.linear (not a raw @) for the adapter matmuls so they are autocast-eligible: under
        # bf16 autocast the fp32 adapter params are cast to match x, exactly like nn.Linear. A raw
        # `@` is NOT covered on MPS and crashes with a bf16-vs-float dtype mismatch (found on the
        # first real run — the fp32/cpu tests never exercised autocast).
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """`W + (alpha/r)·B A` — the effective dense weight, for merge-back."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


@dataclass
class LoRAInfo:
    n_adapted: int  # how many Linear layers were wrapped
    trainable_params: int  # total A+B params (what the optimizer sees)
    total_params: int  # whole model
    targets: tuple[str, ...]
    r: int
    alpha: float


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] | str = "attn",
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInfo:
    """Replace every leaf `nn.Linear` whose attribute name is in `targets` with a `LoRALinear`, and
    freeze all other parameters. Mutates `model` in place. `targets` may be a preset key
    ("attn" | "attn+ffn" | "ffn") or an explicit tuple of leaf names.

    `lm_head` is never adapted here: it's tied to the token embedding (D-016), so wrapping it would
    silently train the embedding too and break weight tying.
    """
    target_names = TARGET_PRESETS[targets] if isinstance(targets, str) else tuple(targets)

    # Freeze everything first; the adapters we create below start out requires_grad=True.
    for p in model.parameters():
        p.requires_grad_(False)

    n_adapted = 0
    for module in model.modules():
        for attr, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr in target_names:
                setattr(module, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                n_adapted += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoRAInfo(
        n_adapted=n_adapted, trainable_params=trainable, total_params=total,
        targets=target_names, r=r, alpha=alpha,
    )


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The trainable adapter parameters (A and B of every `LoRALinear`) — what the optimizer trains."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors, keyed by module path — the tiny artifact you'd actually ship."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return sd


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter tensors produced by `lora_state_dict` into an already-`apply_lora`'d model."""
    by_name = dict(model.named_modules())
    for key, tensor in state.items():
        mod_name, which = key.rsplit(".", 1)
        module = by_name[mod_name]
        target = module.lora_A if which == "lora_A" else module.lora_B
        with torch.no_grad():
            target.copy_(tensor.to(target.device))


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every `LoRALinear` back into a plain `nn.Linear` (weight = `W + (alpha/r)·B A`), in
    place. After this the model is a normal dense model — same weights, no adapter overhead — so
    existing eval/inference code loads it with a plain `state_dict`."""
    for module in model.modules():
        for attr, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.copy_(child.merged_weight())
                base.weight.requires_grad_(True)
                if base.bias is not None:
                    base.bias.requires_grad_(True)
                setattr(module, attr, base)
    return model


def optimizer_state_bytes(n_trainable: int, bytes_per_scalar: int = 4) -> int:
    """AdamW training-time memory attributable to trainable params: gradient + two fp32 moments =
    3 scalars each. The exact quantity LoRA shrinks (the frozen forward weights/activations are
    unchanged)."""
    return 3 * n_trainable * bytes_per_scalar
