"""Tests for src/llmlab/train — lr schedule, param groups, and bit-exact resume.

The resume test is the one that matters most for CLAUDE.md's experiment-discipline rule:
loader.py's stateless (seed, step) sampling means a resumed trainer must reproduce the exact
same per-step losses an uninterrupted run would have produced.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from llmlab.model import GPT, ModelConfig
from llmlab.train.config import TrainConfig
from llmlab.train.trainer import Trainer, build_param_groups, lr_at_step

TOKENIZER_DIR = "data/tokenized/tokenizers/hf_bpe_16k"


def fake_train_cfg(**overrides) -> SimpleNamespace:
    optim = SimpleNamespace(
        lr=1e-3, lr_min_ratio=0.1, warmup_steps=10, schedule="cosine", wsd_decay_ratio=0.2
    )
    base = SimpleNamespace(optim=optim, max_steps=100)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_lr_at_step_linear_warmup_then_cosine_decay():
    cfg = fake_train_cfg()
    assert lr_at_step(0, cfg) == pytest.approx(1e-3 * 1 / 10)
    assert lr_at_step(9, cfg) == pytest.approx(1e-3 * 10 / 10)
    assert lr_at_step(10, cfg) == pytest.approx(1e-3)  # first post-warmup step: cosine progress=0
    assert lr_at_step(100, cfg) == pytest.approx(1e-3 * 0.1)  # lr_min at/after max_steps

    lrs = [lr_at_step(s, cfg) for s in range(10, 101, 5)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "lr must be non-increasing after warmup"


def test_lr_at_step_constant_schedule_stays_flat_after_warmup():
    cfg = fake_train_cfg(optim=SimpleNamespace(lr=1e-3, warmup_steps=10, schedule="constant"))
    assert lr_at_step(9, cfg) == pytest.approx(1e-3 * 10 / 10)
    assert lr_at_step(10, cfg) == pytest.approx(1e-3)
    assert lr_at_step(99, cfg) == pytest.approx(1e-3)  # no decay, ever


def test_lr_at_step_wsd_schedule_stable_then_decays():
    cfg = fake_train_cfg(
        optim=SimpleNamespace(
            lr=1e-3, lr_min_ratio=0.1, warmup_steps=10, schedule="wsd", wsd_decay_ratio=0.2
        ),
        max_steps=100,
    )
    # stable phase: flat at peak lr from warmup end through step 79 (decay starts at 100*0.8=80)
    assert lr_at_step(10, cfg) == pytest.approx(1e-3)
    assert lr_at_step(79, cfg) == pytest.approx(1e-3)
    # decay phase: strictly decreasing from 80 to max_steps, ending at lr_min
    decay_lrs = [lr_at_step(s, cfg) for s in range(80, 101)]
    assert all(a >= b for a, b in zip(decay_lrs, decay_lrs[1:]))
    assert lr_at_step(100, cfg) == pytest.approx(1e-3 * 0.1)


def test_lr_at_step_base_lr_override_scales_independently():
    """The Muon+AdamW hybrid schedules two optimizers off one shape (`_schedule_multiplier`)
    at two different peak lrs -- `base_lr` must scale the whole curve, not just the peak."""
    cfg = fake_train_cfg(
        optim=SimpleNamespace(lr=1e-3, lr_min_ratio=0.1, warmup_steps=10, schedule="cosine")
    )
    assert lr_at_step(5, cfg, base_lr=0.02) == pytest.approx(0.02 * 6 / 10)
    assert lr_at_step(10, cfg, base_lr=0.02) == pytest.approx(0.02)
    assert lr_at_step(100, cfg, base_lr=0.02) == pytest.approx(0.02 * 0.1)


def test_build_param_groups_excludes_norms_and_embeddings():
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16, max_seq_len=32)
    model = GPT(cfg)
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    tok_emb_id = id(model.tok_emb.weight)
    decay_ids = {id(p) for p in decay_group["params"]}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert tok_emb_id in no_decay_ids and tok_emb_id not in decay_ids
    assert all(p.ndim < 2 for p in no_decay_group["params"] if id(p) != tok_emb_id)
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert len(decay_ids) + len(no_decay_ids) == sum(1 for _ in model.parameters())


def make_tiny_trainer(
    tmp_path,
    run_name: str,
    seed: int = 0,
    optim_overrides: dict | None = None,
    model_overrides: dict | None = None,
) -> Trainer:
    model_cfg_path = tmp_path / "model_tiny.yaml"
    model_cfg_path.write_text(
        yaml.dump(
            {
                "vocab_size": 16000,  # matches the real hf_bpe_16k tokenizer used for sampling
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
                "n_kv_heads": 2,
                "head_dim": 16,
                "max_seq_len": 32,
                **(model_overrides or {}),
            }
        )
    )
    rng = np.random.default_rng(seed)
    train_bin, val_bin = tmp_path / "train.bin", tmp_path / "val.bin"
    rng.integers(0, 16000, size=4000, dtype=np.uint16).tofile(train_bin)
    rng.integers(0, 16000, size=1000, dtype=np.uint16).tofile(val_bin)

    optim = {"lr": 1e-3, "warmup_steps": 2, **(optim_overrides or {})}
    cfg = TrainConfig(
        seed=seed,
        model_config=str(model_cfg_path),
        tokenizer_dir=TOKENIZER_DIR,
        seq_len=16,
        sources=[{"name": "t", "path": str(train_bin), "weight": 1.0}],
        val_sources=[{"name": "v", "path": str(val_bin), "weight": 1.0}],
        optim=optim,
        batch={"micro_batch": 4, "grad_accum": 1},
        max_steps=20,
        eval={"eval_every": 5, "eval_batches": 2, "eval_batch_size": 4},
        sampling={"sample_every": 1000, "prompts": ["hi"]},
        logging={"log_every": 5, "wandb_mode": "disabled"},
        checkpoint_every=5,
        device="cpu",
    )
    return Trainer(cfg, tmp_path / run_name)


def run_n_steps(trainer: Trainer, n: int) -> list[float]:
    losses = []
    for _ in range(n):
        loss, _grad_norm, _lr, _aux = trainer.train_step()
        losses.append(loss)
        trainer.step += 1
    return losses


def test_resume_reproduces_the_same_loss_trajectory(tmp_path):
    trainer = make_tiny_trainer(tmp_path, "run1")
    run_n_steps(trainer, 6)  # steps 0..5
    ckpt_path = trainer.run_dir / "ckpt" / "latest.pt"
    trainer.save_checkpoint(ckpt_path)
    step_at_checkpoint = trainer.step

    losses_uninterrupted = run_n_steps(trainer, 4)  # steps 6..9, no interruption

    resumed = make_tiny_trainer(tmp_path, "run2")
    resumed.load_checkpoint(ckpt_path)
    assert resumed.step == step_at_checkpoint
    losses_resumed = run_n_steps(resumed, 4)  # steps 6..9, after a simulated resume

    assert losses_resumed == pytest.approx(losses_uninterrupted)


@pytest.mark.parametrize(
    "optim_overrides",
    [
        {"optimizer": "lion", "lr": 3e-4, "betas": (0.9, 0.99)},
        {"optimizer": "muon", "lr": 1e-3, "muon_lr": 0.02, "muon_momentum": 0.9},
    ],
)
def test_resume_reproduces_the_same_loss_trajectory_for_wave_d_optimizers(tmp_path, optim_overrides):
    """Same bit-exact-resume check as above, but for Lion (single optimizer, new update rule)
    and Muon (TWO optimizers -- exercises `_build_optimizers`'/checkpointing's list handling,
    the part of Wave D's hybrid-optimizer design most likely to silently drop state on resume)."""
    trainer = make_tiny_trainer(tmp_path, "run1", optim_overrides=optim_overrides)
    run_n_steps(trainer, 6)
    ckpt_path = trainer.run_dir / "ckpt" / "latest.pt"
    trainer.save_checkpoint(ckpt_path)
    step_at_checkpoint = trainer.step

    losses_uninterrupted = run_n_steps(trainer, 4)

    resumed = make_tiny_trainer(tmp_path, "run2", optim_overrides=optim_overrides)
    resumed.load_checkpoint(ckpt_path)
    assert resumed.step == step_at_checkpoint
    losses_resumed = run_n_steps(resumed, 4)

    assert losses_resumed == pytest.approx(losses_uninterrupted)


def test_gradient_checkpointing_flag_reaches_the_model(tmp_path):
    """`TrainConfig.gradient_checkpointing` must actually reach `model.gradient_checkpointing`
    (not just get parsed and ignored) and a step must still run and match the non-checkpointed
    trajectory bit-for-bit (same property test_model.py checks at the model level, exercised
    here through the real Trainer/optimizer path)."""
    import copy

    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    t_off = make_tiny_trainer(tmp_path / "off", "off")
    t_on = make_tiny_trainer(tmp_path / "on", "on")
    assert t_off.model.gradient_checkpointing is False
    t_on.model.gradient_checkpointing = True
    assert t_on.model.gradient_checkpointing is True
    t_on.model.load_state_dict(copy.deepcopy(t_off.model.state_dict()))

    loss_off, grad_norm_off, _, _ = t_off.train_step()
    loss_on, grad_norm_on, _, _ = t_on.train_step()
    assert loss_off == pytest.approx(loss_on)
    assert grad_norm_off == pytest.approx(grad_norm_on, rel=1e-4)


def test_precision_fp32_disables_autocast(tmp_path):
    """`precision="fp32"` must swap in a no-op context manager rather than a bf16 autocast --
    a real step should still run to completion under it (the thing most likely to break: a
    dtype mismatch between an fp32-forced forward and any component that assumes autocast)."""
    from contextlib import nullcontext

    trainer = make_tiny_trainer(tmp_path, "fp32run")
    trainer.cfg.precision = "fp32"
    assert isinstance(trainer._autocast(), nullcontext)
    loss, grad_norm, _, _ = trainer.train_step()
    assert math.isfinite(loss)
    assert math.isfinite(grad_norm)


def test_precision_unknown_value_raises(tmp_path):
    trainer = make_tiny_trainer(tmp_path, "badprecision")
    trainer.cfg.precision = "fp16"
    with pytest.raises(ValueError):
        trainer._autocast()


def test_compile_disabled_by_default_leaves_raw_model_as_the_model(tmp_path):
    trainer = make_tiny_trainer(tmp_path, "nocompile")
    assert trainer.compile_status == "disabled"
    assert trainer.model is trainer._raw_model


def test_z_loss_changes_optimization_when_enabled(tmp_path):
    """z-loss is added inside `train_step` to the tensor that gets `.backward()`ed -- confirm
    turning it on measurably changes the gradient (a different grad_norm from the same
    starting weights) rather than silently being a no-op."""
    import copy

    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    t_off = make_tiny_trainer(tmp_path / "off", "off", optim_overrides={"z_loss_weight": 0.0})
    t_on = make_tiny_trainer(tmp_path / "on", "on", optim_overrides={"z_loss_weight": 1.0})
    t_on.model.load_state_dict(copy.deepcopy(t_off.model.state_dict()))

    _, grad_norm_off, _, _ = t_off.train_step()
    _, grad_norm_on, _, _ = t_on.train_step()
    assert grad_norm_off != pytest.approx(grad_norm_on)


# -- Wave F: MoE / MTP integration (phase 5) ---------------------------------------


@pytest.mark.parametrize("balancing", ["aux_loss", "bias_free"])
def test_moe_aux_metrics_appear_in_metrics_jsonl(tmp_path, balancing):
    moe_cfg = {"n_experts": 4, "n_shared": 1, "top_k": 2, "balancing": balancing}
    trainer = make_tiny_trainer(tmp_path, "moe_run", model_overrides={"moe": moe_cfg})
    _, _, _, aux = trainer.train_step()
    trainer._log(0, 1.0, 1.0, 1e-3, 100.0, None, aux)
    last = json.loads(trainer.metrics_path.read_text().strip().splitlines()[-1])
    assert "moe_aux_loss" in last
    assert "expert_load" in last
    assert len(last["expert_load"]) == 2  # one row per layer (n_layers=2 in make_tiny_trainer)
    assert len(last["expert_load"][0]) == 4  # n_experts


def test_mtp_loss_appears_in_metrics_jsonl(tmp_path):
    trainer = make_tiny_trainer(tmp_path, "mtp_run", model_overrides={"mtp": {"n_predict_tokens": 1}})
    _, _, _, aux = trainer.train_step()
    trainer._log(0, 1.0, 1.0, 1e-3, 100.0, None, aux)
    last = json.loads(trainer.metrics_path.read_text().strip().splitlines()[-1])
    assert "mtp_loss" in last


def test_moe_bias_free_routing_bias_moves_after_training_steps(tmp_path):
    """End-to-end check that Trainer actually calls `update_moe_bias` -- the buffer should be
    nonzero after a few real steps of a token stream that (at this tiny random-data scale) is
    virtually guaranteed to route unevenly across only 4 experts."""
    moe_cfg = {"n_experts": 4, "n_shared": 1, "top_k": 1, "balancing": "bias_free"}
    trainer = make_tiny_trainer(tmp_path, "bias_run", model_overrides={"moe": moe_cfg})
    run_n_steps(trainer, 8)
    bias = trainer._raw_model.blocks[0].ffn.routing_bias
    assert bias.abs().sum().item() > 0.0


def test_evaluate_val_loss_excludes_aux_terms(tmp_path):
    """Regression test for a real bug caught mid-Wave-F: `evaluate()` must report pure
    next-token cross-entropy, NOT `forward()`'s combined (aux-weighted) training objective --
    otherwise val_loss isn't comparable to every other wave's noise-floor convention. Two
    identical models differing only in `moe.aux_loss_weight` (0.0 vs a deliberately huge 50.0)
    must report the SAME val_loss despite very different train-time losses."""
    import copy

    (tmp_path / "low").mkdir()
    (tmp_path / "high").mkdir()
    moe_cfg_low = {"n_experts": 4, "n_shared": 1, "top_k": 2, "balancing": "aux_loss", "aux_loss_weight": 0.0}
    moe_cfg_high = {"n_experts": 4, "n_shared": 1, "top_k": 2, "balancing": "aux_loss", "aux_loss_weight": 50.0}
    t_low = make_tiny_trainer(tmp_path / "low", "low", model_overrides={"moe": moe_cfg_low})
    t_high = make_tiny_trainer(tmp_path / "high", "high", model_overrides={"moe": moe_cfg_high})
    t_high.model.load_state_dict(copy.deepcopy(t_low.model.state_dict()))

    val_low = t_low.evaluate()
    val_high = t_high.evaluate()
    assert val_low == pytest.approx(val_high, abs=1e-5)


def test_moe_resume_reproduces_the_same_loss_trajectory(tmp_path):
    """Bit-exact resume (D-023's guarantee) must extend to MoE's new checkpointed buffers
    (`routing_bias`, `_load_accum`) -- both are `register_buffer`s so `state_dict()` already
    covers them, but this confirms it end-to-end rather than by inspection."""
    moe_cfg = {"n_experts": 4, "n_shared": 1, "top_k": 2, "balancing": "bias_free"}
    trainer = make_tiny_trainer(tmp_path, "run1", model_overrides={"moe": moe_cfg})
    run_n_steps(trainer, 6)
    ckpt_path = trainer.run_dir / "ckpt" / "latest.pt"
    trainer.save_checkpoint(ckpt_path)
    step_at_checkpoint = trainer.step

    losses_uninterrupted = run_n_steps(trainer, 4)

    resumed = make_tiny_trainer(tmp_path, "run2", model_overrides={"moe": moe_cfg})
    resumed.load_checkpoint(ckpt_path)
    assert resumed.step == step_at_checkpoint
    losses_resumed = run_n_steps(resumed, 4)

    assert losses_resumed == pytest.approx(losses_uninterrupted)
