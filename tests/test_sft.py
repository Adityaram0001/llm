"""Tests for phase 8 Part A: chat_format (render/encode/mask) + sft_loader (pad/mask/batch).

Uses the REAL hf_bpe_16k tokenizer (the reserved special tokens must exist and be single tokens),
and a tiny GPT with vocab_size=16000 to exercise a masked forward end-to-end.
"""

from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer

from llmlab.data.chat_format import (
    ASSISTANT,
    EOT,
    USER,
    Message,
    describe_example,
    encode_example,
    encode_prompt,
    render_chat,
    to_messages,
)
from llmlab.data.sft_loader import IGNORE_INDEX, SFTDataset
from llmlab.model import GPT, ModelConfig

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


# -- chat_format --------------------------------------------------------


def test_render_single_turn(tokenizer):
    s = render_chat(to_messages("Hi", "Hello there."))
    assert s == f"{USER}Hi{ASSISTANT}Hello there.{EOT}"


def test_render_generation_prompt_ends_on_assistant_cue():
    s = render_chat([Message("user", "Hi")], add_generation_prompt=True)
    assert s == f"{USER}Hi{ASSISTANT}"


def test_render_generation_prompt_rejects_trailing_assistant():
    with pytest.raises(ValueError):
        render_chat(to_messages("Hi", "Hello."), add_generation_prompt=True)


def test_segmentwise_encoding_matches_monolithic(tokenizer):
    """Encoding each turn's content separately + splicing marker IDs must reproduce a single
    encode() of the whole rendered string — otherwise the loss mask wouldn't align with what the
    model actually sees."""
    msgs = to_messages("What is a cat?", "A small domesticated animal.")
    ids, _ = encode_example(tokenizer, msgs)
    mono = tokenizer.encode(render_chat(msgs)).ids
    assert ids == mono


def test_mask_covers_only_assistant_content_and_eot(tokenizer):
    msgs = to_messages("Define cat.", "A pet.")
    ids, sup = encode_example(tokenizer, msgs)
    assert len(ids) == len(sup)
    uid, aid, eid = (tokenizer.token_to_id(t) for t in (USER, ASSISTANT, EOT))
    # user marker + user content + assistant marker are all context (0)
    a_pos = ids.index(aid)
    assert all(s == 0 for s in sup[: a_pos + 1])
    assert ids[0] == uid
    # everything after the assistant marker is supervised, including the final EOT
    assert all(s == 1 for s in sup[a_pos + 1 :])
    assert ids[-1] == eid and sup[-1] == 1


def test_supervise_eot_false_drops_stop_token(tokenizer):
    ids, sup = encode_example(tokenizer, to_messages("Q", "A"), supervise_eot=False)
    assert ids[-1] == tokenizer.token_to_id(EOT)
    assert sup[-1] == 0


def test_encode_prompt_is_context_only_and_ends_on_cue(tokenizer):
    ids = encode_prompt(tokenizer, "What does ephemeral mean?")
    assert ids[0] == tokenizer.token_to_id(USER)
    assert ids[-1] == tokenizer.token_to_id(ASSISTANT)


def test_describe_example_aligns(tokenizer):
    ids, sup = encode_example(tokenizer, to_messages("Q", "A"))
    desc = describe_example(tokenizer, ids, sup)
    assert len(desc) == len(ids)
    assert all(isinstance(tok, str) for tok, _ in desc)


# -- sft_loader --------------------------------------------------------


@pytest.fixture
def rows():
    return [
        {"instruction": "What is a cat?", "response": "A small animal.", "meta": {"word": "cat"}},
        {"instruction": "Define dog.", "response": "A loyal pet that barks.", "meta": {}},
        {"instruction": "What is rain?", "response": "Water falling from clouds.", "meta": {}},
    ]


def test_dataset_builds_examples(tokenizer, rows):
    ds = SFTDataset(rows, tokenizer, max_len=128)
    assert len(ds) == 3
    for ex in ds.examples:
        assert len(ex.ids) == len(ex.supervise)
        assert sum(ex.supervise) > 0


def test_collate_shapes_padding_and_mask(tokenizer, rows):
    ds = SFTDataset(rows, tokenizer, max_len=128)
    x, y = ds.collate([0, 1, 2], DEVICE)
    assert x.shape == y.shape
    assert x.shape[0] == 3
    pad_id = tokenizer.token_to_id("<|pad|>")
    # every row has at least one supervised (non-ignore) target
    for r in range(3):
        assert (y[r] != IGNORE_INDEX).sum() > 0
    # wherever x is padded, the corresponding target must be ignored
    assert torch.all(y[x == pad_id] == IGNORE_INDEX)
    # masking is actually happening (some real-token positions are context-masked too)
    assert (y == IGNORE_INDEX).sum() > (x == pad_id).sum()


def test_masked_targets_equal_shifted_supervised_tokens(tokenizer, rows):
    """The single most important invariant: y at a supervised position equals the *next* input
    token, and every non-supervised / pad position is IGNORE_INDEX."""
    ds = SFTDataset(rows, tokenizer, max_len=128)
    ex = ds.examples[0]
    x, y = ds.collate([0], DEVICE)
    n = len(ex.ids) - 1
    for i in range(n):
        if ex.supervise[i + 1] == 1:
            assert y[0, i].item() == ex.ids[i + 1]
        else:
            assert y[0, i].item() == IGNORE_INDEX


def test_right_truncation_warns_and_drops_empty_mask(tokenizer, capsys):
    # max_len tiny enough that a long response gets fully cut off -> no supervised tokens -> dropped
    rows = [{"instruction": "A very long question about many things here", "response": "x " * 50}]
    ds = SFTDataset(rows, tokenizer, max_len=6)
    assert len(ds) == 0  # dropped: nothing supervised survived truncation
    assert "truncated" in capsys.readouterr().out


def test_epoch_batches_cover_all_examples_deterministically(tokenizer, rows):
    ds = SFTDataset(rows, tokenizer, max_len=128)
    b1 = ds.epoch_batches(batch_size=2, seed=0, epoch=0, device=DEVICE)
    b2 = ds.epoch_batches(batch_size=2, seed=0, epoch=0, device=DEVICE)
    assert len(b1) == 2  # 3 examples, batch_size 2 -> 2 batches
    assert torch.equal(b1[0][0], b2[0][0])  # same (seed, epoch) -> identical shuffle
    b3 = ds.epoch_batches(batch_size=2, seed=0, epoch=1, device=DEVICE)
    # different epoch -> (very likely) different order; at least the call must succeed and cover all
    total = sum(xb.shape[0] for xb, _ in b3)
    assert total == 3


def test_masked_forward_runs_and_backprops(tiny_model, tokenizer, rows):
    """End-to-end: a masked batch produces a finite loss that only the supervised tokens drive."""
    ds = SFTDataset(rows, tokenizer, max_len=64)
    x, y = ds.collate([0, 1], DEVICE)
    _, loss = tiny_model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert tiny_model.tok_emb.weight.grad is not None


def test_from_jsonl_roundtrip(tokenizer, rows, tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = SFTDataset.from_jsonl(p, tokenizer, max_len=128)
    assert len(ds) == 3
