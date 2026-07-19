"""Tests for src/llmlab/train/lora.py (phase 8, Part B). Tiny GPT, cpu."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llmlab.model import GPT, ModelConfig
from llmlab.train.lora import (
    LoRALinear,
    apply_lora,
    load_lora_state,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    optimizer_state_bytes,
)

DEVICE = torch.device("cpu")


def tiny_gpt() -> GPT:
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16, max_seq_len=32
    )
    m = GPT(cfg)
    m.eval()
    return m


def test_lora_linear_is_identity_at_init():
    """B=0 means the adapter contributes exactly 0 at init -> output == frozen base output."""
    base = nn.Linear(16, 24, bias=False)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert torch.allclose(lora(x), base(x))
    assert torch.count_nonzero(lora.lora_B) == 0
    assert torch.count_nonzero(lora.lora_A) > 0  # A is random (nonzero)


def test_apply_lora_freezes_base_and_counts_params():
    m = tiny_gpt()
    info = apply_lora(m, "attn", r=8, alpha=16.0)
    # 2 blocks x 4 attn projections
    assert info.n_adapted == 8
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)  # only adapters train
    # r*(in+out) per q/o proj (d_model x d_model=32) + k/v (d_model x n_kv*head_dim=32) -> all 32x32
    expected = info.n_adapted * 8 * (32 + 32)
    assert info.trainable_params == expected


def test_apply_lora_output_matches_base_at_init():
    m = tiny_gpt()
    idx = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        before = m(idx)[0]
    apply_lora(m, "attn+ffn", r=8, alpha=16.0)
    with torch.no_grad():
        after = m(idx)[0]
    assert torch.allclose(before, after, atol=1e-5)  # adapters are zero at init


def test_attn_ffn_targets_more_than_attn_only():
    a = apply_lora(tiny_gpt(), "attn", r=8)
    af = apply_lora(tiny_gpt(), "attn+ffn", r=8)
    assert af.n_adapted > a.n_adapted
    assert af.trainable_params > a.trainable_params


def test_merge_lora_preserves_output():
    m = tiny_gpt()
    apply_lora(m, "attn+ffn", r=8, alpha=16.0)
    # perturb adapters so the LoRA path is non-trivial
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(std=0.1)
    idx = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        adapted = m(idx)[0]
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())  # all folded away
    with torch.no_grad():
        merged = m(idx)[0]
    assert torch.allclose(adapted, merged, atol=1e-5)


def test_state_dict_roundtrip():
    m1 = tiny_gpt()
    base_state = copy.deepcopy(m1.state_dict())  # snapshot base weights before wrapping
    apply_lora(m1, "attn", r=8)
    for mod in m1.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(std=0.1)
    state = lora_state_dict(m1)
    assert all(k.endswith((".lora_A", ".lora_B")) for k in state)

    m2 = tiny_gpt()
    m2.load_state_dict(base_state)  # give m2 the SAME base weights as m1
    apply_lora(m2, "attn", r=8)
    load_lora_state(m2, state)
    idx = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        assert torch.allclose(m1(idx)[0], m2(idx)[0], atol=1e-6)


def test_gradient_flows_to_adapter():
    m = tiny_gpt()
    apply_lora(m, "attn", r=8)
    idx = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    _, loss = m(idx, y)
    loss.backward()
    # base weights frozen (no grad), at least one adapter B has a real gradient
    a_lora = next(mod for mod in m.modules() if isinstance(mod, LoRALinear))
    assert a_lora.base.weight.grad is None
    assert a_lora.lora_B.grad is not None and torch.count_nonzero(a_lora.lora_B.grad) > 0


def test_lora_parameters_matches_trainable():
    m = tiny_gpt()
    apply_lora(m, "attn+ffn", r=8)
    lp = lora_parameters(m)
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert sum(p.numel() for p in lp) == sum(p.numel() for p in trainable)


def test_optimizer_state_bytes():
    assert optimizer_state_bytes(1000) == 3 * 1000 * 4


def test_forward_under_bf16_autocast():
    """Regression: the adapter matmuls must be autocast-eligible so fp32 A/B don't clash with a
    bf16 base output. A raw `@` crashed on the first real run; F.linear fixes it."""
    base = nn.Linear(16, 24, bias=False)
    lora = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(4, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = lora(x)
    assert torch.isfinite(out).all()
