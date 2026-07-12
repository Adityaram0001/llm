"""Tests for src/llmlab/train — lr schedule, param groups, and bit-exact resume.

The resume test is the one that matters most for CLAUDE.md's experiment-discipline rule:
loader.py's stateless (seed, step) sampling means a resumed trainer must reproduce the exact
same per-step losses an uninterrupted run would have produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from llmlab.model import GPT, ModelConfig
from llmlab.train.config import TrainConfig
from llmlab.train.trainer import Trainer, build_param_groups, lr_at_step

TOKENIZER_DIR = "data/tokenized/tokenizers/hf_bpe_16k"


def fake_train_cfg(**overrides) -> SimpleNamespace:
    optim = SimpleNamespace(lr=1e-3, lr_min_ratio=0.1, warmup_steps=10)
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


def make_tiny_trainer(tmp_path, run_name: str, seed: int = 0) -> Trainer:
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
            }
        )
    )
    rng = np.random.default_rng(seed)
    train_bin, val_bin = tmp_path / "train.bin", tmp_path / "val.bin"
    rng.integers(0, 16000, size=4000, dtype=np.uint16).tofile(train_bin)
    rng.integers(0, 16000, size=1000, dtype=np.uint16).tofile(val_bin)

    cfg = TrainConfig(
        seed=seed,
        model_config=str(model_cfg_path),
        tokenizer_dir=TOKENIZER_DIR,
        seq_len=16,
        sources=[{"name": "t", "path": str(train_bin), "weight": 1.0}],
        val_sources=[{"name": "v", "path": str(val_bin), "weight": 1.0}],
        optim={"lr": 1e-3, "warmup_steps": 2},
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
        loss, _grad_norm, _lr = trainer.train_step()
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
