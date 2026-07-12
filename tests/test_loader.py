"""Tests for src/llmlab/data/loader.py — determinism, mixing, and doc-boundary respecting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llmlab.data.loader import MixedSourceLoader, Source


def write_bin(path, n_tokens: int, vocab: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    arr.tofile(path)
    return arr


def test_single_source_shapes_and_next_token_property(tmp_path):
    bin_path = tmp_path / "train.bin"
    write_bin(bin_path, n_tokens=2000)
    src = Source(name="only", bin_path=bin_path, weight=1.0)
    loader = MixedSourceLoader([src], seq_len=16, seed=0)

    x, y = loader.get_batch(step=0, batch_size=4, device=torch.device("cpu"))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y is x shifted by one token within the same window
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_is_deterministic_given_seed_and_step(tmp_path):
    bin_path = tmp_path / "train.bin"
    write_bin(bin_path, n_tokens=2000)
    src = Source(name="only", bin_path=bin_path, weight=1.0)
    loader = MixedSourceLoader([src], seq_len=16, seed=42)

    x1, y1 = loader.get_batch(step=7, batch_size=8, device=torch.device("cpu"))
    x2, y2 = loader.get_batch(step=7, batch_size=8, device=torch.device("cpu"))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)

    # a fresh loader instance with the same seed reproduces the same step exactly (resume)
    loader2 = MixedSourceLoader([src], seq_len=16, seed=42)
    x3, _ = loader2.get_batch(step=7, batch_size=8, device=torch.device("cpu"))
    assert torch.equal(x1, x3)

    # a different step gives a different batch (not a hash collision fluke)
    x4, _ = loader.get_batch(step=8, batch_size=8, device=torch.device("cpu"))
    assert not torch.equal(x1, x4)


def test_mixing_weights_respected_over_many_batches(tmp_path):
    # disjoint value ranges so a sampled window's source is identifiable by value alone
    a_path, b_path = tmp_path / "a.bin", tmp_path / "b.bin"
    rng = np.random.default_rng(1)
    rng.integers(0, 50, size=5000).astype(np.uint16).tofile(a_path)  # values in [0, 50)
    rng.integers(1000, 1050, size=5000).astype(np.uint16).tofile(b_path)  # values in [1000, 1050)
    src_a = Source(name="a", bin_path=a_path, weight=3.0)
    src_b = Source(name="b", bin_path=b_path, weight=1.0)
    loader = MixedSourceLoader([src_a, src_b], seq_len=8, seed=0)

    from_a = 0
    total = 0
    for step in range(200):
        x, _ = loader.get_batch(step=step, batch_size=4, device=torch.device("cpu"))
        from_a += (x.numpy() < 1000).all(axis=1).sum()
        total += x.shape[0]
    assert 0.65 <= from_a / total <= 0.85  # weight 3:1 -> ~0.75 expected


def test_respect_doc_boundaries_windows_stay_within_one_document(tmp_path):
    # 3 documents of lengths 5, 50, 5 tokens -> only the middle doc has room for seq_len=16
    doc_lens = [5, 50, 5]
    rng = np.random.default_rng(0)
    tokens = np.concatenate([rng.integers(0, 100, size=n, dtype=np.uint16) for n in doc_lens])
    bin_path = tmp_path / "docs.bin"
    tokens.tofile(bin_path)
    docstarts_path = tmp_path / "docs_docstarts.npy"
    np.save(docstarts_path, np.array([0, 5, 55], dtype=np.int64))

    src = Source(
        name="docs",
        bin_path=bin_path,
        weight=1.0,
        respect_doc_boundaries=True,
        docstarts_path=docstarts_path,
    )
    loader = MixedSourceLoader([src], seq_len=16, seed=0)
    middle_doc = tokens[5:55].astype(np.int64)
    for step in range(50):
        x, _ = loader.get_batch(step=step, batch_size=8, device=torch.device("cpu"))
        for row in x.numpy():
            windows = np.lib.stride_tricks.sliding_window_view(middle_doc, len(row))
            assert (windows == row).all(axis=1).any(), "window strayed outside the valid document"


def test_source_requires_docstarts_when_respecting_boundaries(tmp_path):
    with pytest.raises(ValueError):
        Source(name="bad", bin_path=tmp_path / "x.bin", weight=1.0, respect_doc_boundaries=True)


def test_all_weights_must_be_positive(tmp_path):
    bin_path = tmp_path / "train.bin"
    write_bin(bin_path, n_tokens=2000)
    src = Source(name="only", bin_path=bin_path, weight=0.0)
    with pytest.raises(ValueError):
        MixedSourceLoader([src], seq_len=16, seed=0)


def test_fixed_eval_batches_stable_across_calls(tmp_path):
    bin_path = tmp_path / "val.bin"
    write_bin(bin_path, n_tokens=2000)
    src = Source(name="val", bin_path=bin_path, weight=1.0)
    loader = MixedSourceLoader([src], seq_len=16, seed=123)

    batches1 = loader.fixed_eval_batches(n_batches=3, batch_size=2, device=torch.device("cpu"))
    batches2 = loader.fixed_eval_batches(n_batches=3, batch_size=2, device=torch.device("cpu"))
    for (x1, y1), (x2, y2) in zip(batches1, batches2):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
