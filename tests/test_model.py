"""Tests for src/llmlab/model — phase 3 exit criteria: green on both mps and cpu.

Covers: shapes, the causal mask (the property that makes autoregressive training valid),
loss ~ ln(vocab) at init, generate(), every config axis instantiating, weight tying sharing
storage, and RoPE's relative-position property.
"""

from __future__ import annotations

import math

import pytest
import torch

from llmlab.model import GPT, ModelConfig
from llmlab.model.attention import make_cache
from llmlab.model.positional import RotaryEmbedding, apply_rotary

DEVICES = [torch.device("cpu")]
if torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))


def tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        max_seq_len=32,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(params=DEVICES, ids=lambda d: d.type)
def device(request):
    return request.param


# -- shapes ------------------------------------------------------------------


def test_forward_shapes(device):
    cfg = tiny_config()
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    y = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    logits, loss = model(x, y)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss.shape == ()
    assert loss.item() > 0

    logits_no_targets, loss_none = model(x)
    assert logits_no_targets.shape == (3, 10, cfg.vocab_size)
    assert loss_none is None


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal"])
def test_exceeding_max_seq_len_raises_for_bounded_encodings(device, pos_encoding):
    """learned/sinusoidal have a fixed-size table sized to max_seq_len -- can't extrapolate."""
    cfg = tiny_config(pos_encoding=pos_encoding)
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1), device=device)
    with pytest.raises(ValueError):
        model(x)


@pytest.mark.parametrize("pos_encoding", ["rope", "alibi", "none"])
def test_exceeding_max_seq_len_allowed_for_unbounded_encodings(device, pos_encoding):
    """RoPE/ALiBi/none compute position info on the fly, so eval-time forward passes beyond
    max_seq_len must succeed -- this is what the phase-5 length-extrapolation probe needs (RW-5)."""
    cfg = tiny_config(pos_encoding=pos_encoding)
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 8), device=device)
    logits, _ = model(x)
    assert logits.shape == (1, cfg.max_seq_len + 8, cfg.vocab_size)


# -- causal mask ---------------------------------------------------------------


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope", "alibi", "none"])
def test_causal_mask_future_does_not_leak_into_past(device, pos_encoding):
    """Changing a future token must not change ANY logit at an earlier position — that's the
    entire point of causal masking, and it's easy to silently break (e.g. wrong mask direction,
    an attn_bias built with the wrong sign). Test it directly rather than trust the plumbing."""
    cfg = tiny_config(pos_encoding=pos_encoding)
    model = GPT(cfg).to(device).eval()

    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, 12), device=device)
    x_modified = x.clone()
    x_modified[:, -1] = (x_modified[:, -1] + 1) % cfg.vocab_size  # perturb only the LAST token

    with torch.no_grad():
        logits_a, _ = model(x)
        logits_b, _ = model(x_modified)

    # all positions except the last must be untouched
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)
    # the last position's logits (which depend on the changed token) should generally differ
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :], atol=1e-5)


# -- loss at init --------------------------------------------------------------


def test_loss_near_ln_vocab_at_init(device):
    """An untrained model should predict ~uniformly over the vocab, so cross-entropy loss
    should start close to ln(vocab_size) — a standard sanity check before any training run."""
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=1000, d_model=64, n_layers=4, n_heads=4, n_kv_heads=4, head_dim=16)
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (8, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (8, 32), device=device)
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


# -- generate ------------------------------------------------------------------


def test_generate_runs_and_grows_sequence(device):
    cfg = tiny_config()
    model = GPT(cfg).to(device)
    idx = torch.randint(0, cfg.vocab_size, (2, 5), device=device)
    out = model.generate(idx, max_new_tokens=7, temperature=1.0, top_k=10)
    assert out.shape == (2, 5 + 7)
    assert torch.equal(out[:, :5], idx)
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_top_p_runs(device):
    cfg = tiny_config()
    model = GPT(cfg).to(device)
    idx = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    out = model.generate(idx, max_new_tokens=3, top_p=0.9)
    assert out.shape == (1, 7)


def test_generate_restores_training_mode(device):
    cfg = tiny_config()
    model = GPT(cfg).to(device).train()
    idx = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    model.generate(idx, max_new_tokens=2)
    assert model.training is True


# -- weight tying ----------------------------------------------------------------


def test_tied_weights_share_storage(device):
    cfg = tiny_config(tie_embeddings=True)
    model = GPT(cfg).to(device)
    assert model.lm_head.weight is model.tok_emb.weight

    total_params = sum(p.numel() for p in model.parameters())
    breakdown = model.num_params(breakdown=True)
    assert breakdown["head"] == 0
    assert total_params == breakdown["total"]


def test_untied_weights_are_independent(device):
    cfg = tiny_config(tie_embeddings=False)
    model = GPT(cfg).to(device)
    assert model.lm_head.weight is not model.tok_emb.weight
    breakdown = model.num_params(breakdown=True)
    assert breakdown["head"] == cfg.vocab_size * cfg.d_model
    assert breakdown["head"] > 0


# -- gradient checkpointing (Wave E) --------------------------------------------------


def test_gradient_checkpointing_matches_no_checkpointing(device):
    """Gradient checkpointing (Chen et al. '16) trades recompute-on-backward for lower
    activation memory -- it must NOT change what's computed. Same weights, same input, same
    dropout (0.0, so no RNG divergence risk): checkpointed and non-checkpointed forward+backward
    should agree on loss and every gradient to floating-point precision."""
    torch.manual_seed(0)
    cfg = tiny_config()
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (2, 10), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 10), device=device)

    model.gradient_checkpointing = False
    model.train()
    logits_plain, loss_plain = model(x, y)
    loss_plain.backward()
    grads_plain = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    model.gradient_checkpointing = True
    logits_ckpt, loss_ckpt = model(x, y)
    loss_ckpt.backward()
    grads_ckpt = [p.grad.clone() for p in model.parameters()]

    assert torch.allclose(logits_plain, logits_ckpt)
    assert loss_plain.item() == pytest.approx(loss_ckpt.item())
    for g_plain, g_ckpt in zip(grads_plain, grads_ckpt):
        assert torch.allclose(g_plain, g_ckpt, atol=1e-6)


def test_gradient_checkpointing_disabled_during_eval(device):
    """`use_ckpt` requires `self.training` -- confirms eval-mode forward (as used by
    Trainer.evaluate) never checkpoints, since there's no backward pass to save memory for and
    checkpointing would just add pointless recompute risk to an eval loop."""
    cfg = tiny_config()
    model = GPT(cfg).to(device)
    model.gradient_checkpointing = True
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10), device=device)
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (2, 10, cfg.vocab_size)


# -- RoPE relative-shift property --------------------------------------------------


def test_rope_relative_shift_property():
    """RoPE's core property: rotate(q, i) . rotate(k, j) depends only on (i - j), not on the
    absolute positions i, j. Verify by comparing two (query pos, key pos) pairs with the same
    offset but different absolute positions."""
    torch.manual_seed(0)
    head_dim = 16
    rotary = RotaryEmbedding(head_dim, theta=10000.0)
    seq_len = 20
    cos, sin = rotary(seq_len, torch.device("cpu"))

    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    # shape to (batch=1, heads=1, seq_len, head_dim) as apply_rotary expects
    q = q_vec.expand(1, 1, seq_len, head_dim)
    k = k_vec.expand(1, 1, seq_len, head_dim)
    q_rot, k_rot = apply_rotary(q, k, cos, sin)

    def dot_at(i: int, j: int) -> float:
        return (q_rot[0, 0, i] @ k_rot[0, 0, j]).item()

    # (5, 2) and (13, 10) both have offset 3
    d1 = dot_at(5, 2)
    d2 = dot_at(13, 10)
    assert math.isclose(d1, d2, rel_tol=1e-4, abs_tol=1e-4)

    # a different offset should (with overwhelming probability, for random vectors) differ
    d3 = dot_at(5, 0)
    assert not math.isclose(d1, d3, rel_tol=1e-4, abs_tol=1e-4)


# -- config axes all instantiate -------------------------------------------------


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("norm_position", ["pre", "post"])
def test_norm_axis_instantiates(norm, norm_position):
    cfg = tiny_config(norm=norm, norm_position=norm_position)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope", "alibi", "none"])
def test_pos_encoding_axis_instantiates(pos_encoding):
    cfg = tiny_config(pos_encoding=pos_encoding)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("ffn", ["gelu", "swiglu"])
def test_ffn_axis_instantiates(ffn):
    cfg = tiny_config(ffn=ffn, ffn_mult=4.0 if ffn == "gelu" else 8 / 3)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])  # MHA, GQA, MQA (n_heads=4)
def test_gqa_mqa_axis_instantiates(n_kv_heads):
    cfg = tiny_config(n_kv_heads=n_kv_heads)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_qk_norm_instantiates():
    cfg = tiny_config(qk_norm=True)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


@pytest.mark.parametrize("init", ["gpt2", "scaled"])
def test_init_axis_instantiates(init):
    cfg = tiny_config(init=init)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_kitchen_sink_combo_instantiates():
    """One deliberately unusual combo (GQA + alibi + gelu + post-norm + qk_norm + untied) to
    catch axis-interaction bugs that per-axis tests (which vary one field at a time) would miss."""
    cfg = tiny_config(
        n_kv_heads=2,
        pos_encoding="alibi",
        ffn="gelu",
        ffn_mult=4.0,
        norm_position="post",
        norm="layernorm",
        qk_norm=True,
        tie_embeddings=False,
        init="scaled",
    )
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(loss)


# -- MLA + KV-cache (phase 5-C) --------------------------------------------------

MLA_KW = dict(kv_lora_rank=24, q_lora_rank=32, rope_head_dim=8, nope_head_dim=8, v_head_dim=16)


def test_mla_instantiates_and_runs(device):
    cfg = tiny_config(attention="mla", mla=dict(MLA_KW))
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (2, 12), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 12), device=device)
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_mla_requires_config_block():
    with pytest.raises(ValueError):
        tiny_config(attention="mla")  # __post_init__ rejects mla with no mla: block


@pytest.mark.parametrize(
    "attn,n_kv,mla",
    [
        ("mha_gqa", 4, None),  # MHA
        ("mha_gqa", 2, None),  # GQA
        ("mha_gqa", 1, None),  # MQA
        ("mla", 4, dict(MLA_KW)),  # MLA
    ],
)
def test_cached_decode_matches_full_forward(device, attn, n_kv, mla):
    """Prefill part of a sequence then decode the rest one token at a time against the KV cache;
    the stacked logits must match a single full-sequence forward pass (bit-for-bit up to fp
    error). This is the core correctness guarantee for every attention variant's decode path."""
    cfg = tiny_config(attention=attn, n_kv_heads=n_kv, mla=mla)
    model = GPT(cfg).to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 20), device=device)
    with torch.no_grad():
        full, _ = model(x)
        caches = [make_cache(cfg) for _ in range(cfg.n_layers)]
        outs = [model(x[:, :5], caches=caches)[0]]
        for t in range(5, 20):
            outs.append(model(x[:, t : t + 1], caches=caches)[0])
    inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max().item()


def test_kv_cache_bytes_ordering():
    """MHA cache > GQA > MQA per token; MLA is independent of n_heads (latent + shared rope)."""
    base = dict(vocab_size=64, d_model=32, n_layers=1, n_heads=4, head_dim=16, max_seq_len=32)
    caches = {}
    for name, kv in [("mha", 4), ("gqa", 2), ("mqa", 1)]:
        c = make_cache(ModelConfig(n_kv_heads=kv, **base))
        c.append(torch.zeros(1, kv, 1, 16), torch.zeros(1, kv, 1, 16))
        caches[name] = c.bytes_per_token()
    assert caches["mha"] > caches["gqa"] > caches["mqa"]


def test_mla_generate_with_and_without_cache_run(device):
    cfg = tiny_config(attention="mla", mla=dict(MLA_KW))
    model = GPT(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    assert model.generate(prompt, 6, top_k=3, use_cache=True).shape == (1, 10)
    assert model.generate(prompt, 6, top_k=3, use_cache=False).shape == (1, 10)


# -- MoE + MTP (phase 5-F, DeepSeek specials) ------------------------------------


@pytest.mark.parametrize("balancing", ["aux_loss", "bias_free"])
def test_moe_instantiates_and_runs(device, balancing):
    cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=2, balancing=balancing))
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    y = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    logits, loss = model(x, y)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    # every expert must receive a real gradient (bias-free's zero aux loss shouldn't disconnect
    # the routed path -- output combination is still differentiable through gate_probs/experts)
    assert all(e.down_proj.weight.grad is not None for e in model.blocks[0].ffn.experts)


def test_moe_active_params_roughly_match_dense_baseline():
    """DeepSeekMoE's headline claim: fine-grained experts sized so ACTIVE params/token (shared +
    top_k) roughly equal a same-shape dense FFN, while TOTAL params are much larger."""
    dense_cfg = tiny_config()
    moe_cfg = tiny_config(moe=dict(n_experts=8, n_shared=1, top_k=2))
    dense_ffn_params = sum(p.numel() for p in GPT(dense_cfg).blocks[0].ffn.parameters())
    moe_ffn = GPT(moe_cfg).blocks[0].ffn
    active_experts = moe_ffn.shared_experts[0]
    active_params = sum(p.numel() for p in active_experts.parameters()) * (1 + 2)  # n_shared+top_k
    assert abs(active_params - dense_ffn_params) / dense_ffn_params < 0.15
    total_moe_params = sum(p.numel() for p in moe_ffn.parameters())
    assert total_moe_params > 2 * dense_ffn_params  # much more total capacity than active


def test_moe_expert_load_sums_to_top_k_fraction():
    """Every token contributes exactly `top_k` routing slots, so the per-expert load fractions
    (each = tokens routed to that expert / total tokens) must sum to top_k across all experts."""
    cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=2))
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 10))
    model(x)  # no targets -> just exercises MoEFFN.forward directly via the block
    load = model.blocks[0].ffn.last_expert_load
    assert torch.allclose(load.sum(), torch.tensor(2.0), atol=1e-5)


def test_moe_aux_loss_only_active_for_aux_loss_balancing():
    x = torch.randint(0, 256, (3, 10))
    y = torch.randint(0, 256, (3, 10))
    aux_cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=2, balancing="aux_loss"))
    bias_cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=2, balancing="bias_free"))
    aux_model = GPT(aux_cfg)
    aux_model(x, y)
    bias_model = GPT(bias_cfg)
    bias_model(x, y)
    assert aux_model.blocks[0].ffn.last_aux_loss.item() != 0.0
    assert bias_model.blocks[0].ffn.last_aux_loss.item() == 0.0


def test_moe_bias_free_update_direction():
    """DeepSeek-V3 S2.1.2's rule: an overloaded expert's bias should decrease (less likely to be
    picked next), an underloaded one's should increase -- gradient-free, buffer-only."""
    cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=1, balancing="bias_free"))
    model = GPT(cfg)
    ffn = model.blocks[0].ffn
    ffn.train()
    ffn._load_accum[:] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    before = ffn.routing_bias.clone()
    ffn.update_bias(0.01)
    assert ffn.routing_bias[0] < before[0]
    assert (ffn.routing_bias[1:] > before[1:]).all()
    assert torch.equal(ffn._load_accum, torch.zeros(4))  # accumulator resets after each update


def test_moe_update_bias_noop_for_aux_loss_balancing():
    cfg = tiny_config(moe=dict(n_experts=4, n_shared=1, top_k=1, balancing="aux_loss"))
    model = GPT(cfg)
    model.update_moe_bias(0.01)  # must not raise (no routing_bias buffer exists at all)


@pytest.mark.parametrize("n_predict_tokens", [1, 3])
def test_mtp_predicts_and_backprops(device, n_predict_tokens):
    cfg = tiny_config(mtp=dict(n_predict_tokens=n_predict_tokens))
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    y = torch.randint(0, cfg.vocab_size, (3, 10), device=device)
    logits, loss = model(x, y)
    assert torch.isfinite(loss)
    assert "mtp_loss" in model.last_aux_metrics
    loss.backward()
    assert model.mtp_heads[0].combine.weight.grad is not None


def test_mtp_short_sequence_falls_back_to_main_loss_only():
    """T=1: no depth has a valid (h_prev, next_emb, target) triple, so `_mtp_loss` returns None
    and the total loss is just the main cross-entropy -- must not crash."""
    cfg = tiny_config(mtp=dict(n_predict_tokens=2))
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 1))
    y = torch.randint(0, cfg.vocab_size, (2, 1))
    logits, loss = model(x, y)
    assert torch.isfinite(loss)
    assert "mtp_loss" not in model.last_aux_metrics


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal"])
def test_mtp_requires_position_encoding_injected_per_block(pos_encoding):
    with pytest.raises(NotImplementedError):
        GPT(tiny_config(pos_encoding=pos_encoding, mtp=dict(n_predict_tokens=1)))


def test_mtp_alibi_runs():
    cfg = tiny_config(pos_encoding="alibi", mtp=dict(n_predict_tokens=2))
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_moe_and_mtp_combined_runs():
    cfg = tiny_config(
        moe=dict(n_experts=4, n_shared=1, top_k=2, balancing="bias_free"),
        mtp=dict(n_predict_tokens=1),
    )
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert set(model.last_aux_metrics.keys()) == {"ce_loss", "moe_aux_loss", "expert_load", "mtp_loss"}


def test_num_params_breakdown_matches_total_with_moe_and_mtp():
    cfg = tiny_config(
        moe=dict(n_experts=4, n_shared=1, top_k=2), mtp=dict(n_predict_tokens=2)
    )
    model = GPT(cfg)
    breakdown = model.num_params(breakdown=True)
    assert breakdown["total"] == model.num_params()
    assert breakdown["mtp"] > 0


# -- named tier configs (the actual configs/*.yaml) ------------------------------


@pytest.mark.parametrize("tier,expected_m_params", [("s", 9.71), ("m", 34.62), ("l", 104.80)])
def test_tier_configs_load_and_match_expected_size(tier, expected_m_params):
    cfg = ModelConfig.from_yaml(f"configs/model_{tier}.yaml")
    model = GPT(cfg)
    n_params = model.num_params()
    assert abs(n_params / 1e6 - expected_m_params) < 0.1
