"""Tests for phase 8 Part C: dpo_loader (paired pad/mask/batch) + dpo (sequence_logprobs, loss).

Uses the REAL hf_bpe_16k tokenizer (same convention as test_sft.py) and a tiny GPT to exercise
the paired forward pass end-to-end.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from llmlab.data.dpo_loader import DPODataset
from llmlab.data.sft_loader import IGNORE_INDEX
from llmlab.model import GPT, ModelConfig
from llmlab.train.dpo import dpo_loss, sequence_logprobs
from tokenizers import Tokenizer

TOKENIZER_DIR = "data/tokenized/tokenizers/hf_bpe_16k"
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(f"{TOKENIZER_DIR}/tokenizer.json")


@pytest.fixture(scope="module")
def tiny_model() -> GPT:
    cfg = ModelConfig(
        vocab_size=16000, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16, max_seq_len=64
    )
    model = GPT(cfg)
    model.eval()
    return model


@pytest.fixture
def rows():
    return [
        {
            "instruction": "What does 'cat' mean?",
            "chosen": "A cat is a small domesticated animal that says meow.",
            "rejected": "A cat is a large aquatic mammal that lives underwater.",
            "meta": {"word": "cat", "failure_mode": "wrong_fact"},
        },
        {
            "instruction": "Define 'dog'.",
            "chosen": "A loyal pet that barks.",
            "rejected": "Well, that's an interesting question, let me think about many unrelated things first before I possibly get to an answer eventually.",
            "meta": {"word": "dog", "failure_mode": "verbose"},
        },
        {
            "instruction": "What is 'rain'?",
            "chosen": "Water falling from clouds.",
            "rejected": "The sun was setting over the hills as the travelers made camp for the night.",
            "meta": {"word": "rain", "failure_mode": "off_format"},
        },
    ]


# -- DPODataset --------------------------------------------------------


def test_dataset_builds_paired_examples(tokenizer, rows):
    ds = DPODataset(rows, tokenizer, max_len=128)
    assert len(ds) == 3
    for ex in ds.examples:
        assert sum(ex.chosen_supervise) > 0
        assert sum(ex.rejected_supervise) > 0


def test_collate_returns_four_independently_padded_tensors(tokenizer, rows):
    ds = DPODataset(rows, tokenizer, max_len=128)
    x_c, y_c, x_r, y_r = ds.collate([0, 1, 2], DEVICE)
    assert x_c.shape == y_c.shape
    assert x_r.shape == y_r.shape
    assert x_c.shape[0] == x_r.shape[0] == 3
    # rejected includes a much longer (verbose) response -> its padded width should be >= chosen's
    assert x_r.shape[1] >= x_c.shape[1]
    pad_id = tokenizer.token_to_id("<|pad|>")
    assert torch.all(y_c[x_c == pad_id] == IGNORE_INDEX)
    assert torch.all(y_r[x_r == pad_id] == IGNORE_INDEX)


def test_chosen_and_rejected_share_the_same_instruction_prefix(tokenizer, rows):
    """Both sides of a triple must encode the SAME instruction -- otherwise the preference
    comparison wouldn't be conditioned on the same prompt."""
    ds = DPODataset(rows, tokenizer, max_len=128)
    ex = ds.examples[0]
    from llmlab.data.chat_format import ASSISTANT

    aid = tokenizer.token_to_id(ASSISTANT)
    c_prefix = ex.chosen_ids[: ex.chosen_ids.index(aid) + 1]
    r_prefix = ex.rejected_ids[: ex.rejected_ids.index(aid) + 1]
    assert c_prefix == r_prefix


def test_from_jsonl_roundtrip(tokenizer, rows, tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = DPODataset.from_jsonl(p, tokenizer, max_len=128)
    assert len(ds) == 3


def test_epoch_batches_cover_all_examples_deterministically(tokenizer, rows):
    ds = DPODataset(rows, tokenizer, max_len=128)
    b1 = ds.epoch_batches(batch_size=2, seed=0, epoch=0, device=DEVICE)
    b2 = ds.epoch_batches(batch_size=2, seed=0, epoch=0, device=DEVICE)
    assert len(b1) == 2
    assert torch.equal(b1[0][0], b2[0][0])
    total = sum(xc.shape[0] for xc, _, _, _ in b1)
    assert total == 3


# -- sequence_logprobs / dpo_loss --------------------------------------------------------


def test_sequence_logprobs_matches_manual_gather(tiny_model, tokenizer, rows):
    ds = DPODataset(rows, tokenizer, max_len=64)
    x, y, _, _ = ds.collate([0], DEVICE)
    lp = sequence_logprobs(tiny_model, x, y)
    assert lp.shape == (1,)

    logits, _ = tiny_model(x)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    mask = y != IGNORE_INDEX
    expected = 0.0
    for t in range(y.shape[1]):
        if mask[0, t]:
            expected += log_probs[0, t, y[0, t]].item()
    assert lp.item() == pytest.approx(expected, abs=1e-4)


def test_dpo_loss_at_zero_logratio_equals_log_2():
    """If policy == reference exactly (no drift on either side), reward_chosen ==
    reward_rejected == 0, so loss = -log(sigmoid(0)) = log(2) -- the DPO loss's uninformed prior."""
    z = torch.zeros(4)
    loss, metrics = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)
    assert metrics["reward_margin"] == pytest.approx(0.0)
    assert metrics["reward_accuracy"] == pytest.approx(0.0)  # chosen == rejected, not strictly >


def test_dpo_loss_decreases_as_policy_prefers_chosen_more():
    """Holding the reference fixed, pushing the policy's chosen log-prob up (relative to
    rejected) must monotonically lower the loss -- this is the one thing the whole objective is
    for."""
    ref_c = ref_r = torch.zeros(1)
    losses = []
    for delta in (0.0, 1.0, 3.0, 6.0):
        policy_c = torch.tensor([delta])
        policy_r = torch.zeros(1)
        loss, _ = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.5)
        losses.append(loss.item())
    assert losses == sorted(losses, reverse=True)  # strictly decreasing as delta grows


def test_reward_accuracy_reflects_which_side_the_policy_prefers():
    policy_c = torch.tensor([2.0, -2.0])
    policy_r = torch.tensor([0.0, 0.0])
    ref_c = ref_r = torch.zeros(2)
    _, metrics = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    assert metrics["reward_accuracy"] == pytest.approx(0.5)  # row 0 prefers chosen, row 1 doesn't


def test_paired_forward_backprops_through_policy_only(tiny_model, tokenizer, rows):
    ds = DPODataset(rows, tokenizer, max_len=64)
    x_c, y_c, x_r, y_r = ds.collate([0, 1], DEVICE)
    tiny_model.train()

    ref = GPT(tiny_model.cfg)
    ref.load_state_dict(tiny_model.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    policy_c = sequence_logprobs(tiny_model, x_c, y_c)
    policy_r = sequence_logprobs(tiny_model, x_r, y_r)
    with torch.no_grad():
        ref_c = sequence_logprobs(ref, x_c, y_c)
        ref_r = sequence_logprobs(ref, x_r, y_r)
    loss, metrics = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    assert torch.isfinite(loss)
    # policy == reference at init -> zero log-ratio on both sides -> log(2) loss
    assert loss.item() == pytest.approx(math.log(2), abs=1e-4)
    loss.backward()
    assert tiny_model.tok_emb.weight.grad is not None
    assert ref.tok_emb.weight.grad is None
