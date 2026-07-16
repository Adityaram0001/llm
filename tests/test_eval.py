"""Tests for src/llmlab/eval (phase 6). Uses a tiny GPT sized down from the real S-tier config
but with the REAL hf_bpe_16k tokenizer (vocab_size=16000) — every probe encodes actual English
text through it, so a mismatched toy vocab would break index bounds, not just be slow.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from llmlab.eval import benchmarks, dictionary_probes, domain_probes, generation, perplexity
from llmlab.eval.scoring import encode_prompt_continuation, greedy_continuation, mc_by_loglik, score_continuation
from llmlab.model import GPT, ModelConfig

TOKENIZER_DIR = "data/tokenized/tokenizers/hf_bpe_16k"


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


DEVICE = torch.device("cpu")


def test_encode_prompt_continuation_splits_at_the_boundary(tokenizer):
    prompt_ids, cont_ids = encode_prompt_continuation(tokenizer, "The cat sat on the", " mat.")
    full_ids = tokenizer.encode("The cat sat on the mat.").ids
    assert prompt_ids + cont_ids == full_ids


def test_score_continuation_returns_consistent_sum_and_mean(tiny_model, tokenizer):
    prompt_ids, cont_ids = encode_prompt_continuation(tokenizer, "hello", " world")
    sum_lp, mean_lp, n = score_continuation(tiny_model, prompt_ids, cont_ids, DEVICE)
    assert n == len(cont_ids)
    assert sum_lp == pytest.approx(mean_lp * n, rel=1e-4)


def test_score_continuation_truncates_long_prompt_from_the_left(tiny_model, tokenizer):
    """A prompt longer than max_seq_len - len(continuation) must not crash -- it should drop
    the earliest prompt tokens (same convention as generate())."""
    long_prompt_ids = list(range(1, 200))  # far longer than max_seq_len=64
    cont_ids = [5, 6, 7]
    sum_lp, mean_lp, n = score_continuation(tiny_model, long_prompt_ids, cont_ids, DEVICE)
    assert n == 3


def test_mc_by_loglik_picks_the_argmax_of_its_own_scores(tiny_model, tokenizer):
    prompt_ids = tokenizer.encode("The word means:").ids
    choices = [tokenizer.encode(t).ids for t in [" apple", " banana", " a very long unlikely phrase indeed"]]
    pred_idx, scores = mc_by_loglik(tiny_model, prompt_ids, choices, DEVICE)
    assert pred_idx == max(range(len(scores)), key=lambda i: scores[i])
    assert len(scores) == 3


def test_greedy_continuation_is_deterministic(tiny_model, tokenizer):
    prompt_ids = tokenizer.encode("Once upon a time").ids
    a = greedy_continuation(tiny_model, prompt_ids, 5, DEVICE)
    b = greedy_continuation(tiny_model, prompt_ids, 5, DEVICE)
    assert a == b
    assert len(a) == 5


def test_perplexity_evaluate_split_on_a_synthetic_bin(tiny_model, tokenizer, tmp_path):
    rng = np.random.default_rng(0)
    bin_path = tmp_path / "toy_val.bin"
    rng.integers(0, 16000, size=2000, dtype=np.uint16).tofile(bin_path)
    result = perplexity.evaluate_split(tiny_model, tokenizer, str(bin_path), DEVICE, seq_len=32, batch_size=4)
    assert result["ppl"] > 1.0
    assert result["n_tokens"] > 0
    assert result["n_bytes"] > 0
    assert result["bits_per_byte"] > 0


def test_dictionary_probes_run_end_to_end(tiny_model, tokenizer, tmp_path):
    entries = [
        {"word": "cat", "pos": "n.", "definitions": ["a small domesticated carnivorous mammal"]},
        {"word": "dog", "pos": "n.", "definitions": ["a domesticated carnivorous mammal that barks"]},
        {"word": "fish", "pos": "n.", "definitions": ["a limbless cold-blooded animal with gills"]},
        {"word": "bird", "pos": "n.", "definitions": ["a warm-blooded egg-laying vertebrate with wings"]},
        {"word": "bee", "pos": "n.", "definitions": ["a stinging winged insect that collects pollen"]},
    ]
    val_path = tmp_path / "dictionary.jsonl"
    val_path.write_text("\n".join(json.dumps(e) for e in entries))

    result = dictionary_probes.run(tiny_model, tokenizer, val_path, DEVICE, max_examples=5, seed=0)
    assert result["n_examples"] == 5
    assert result["definition_completion_ppl"] > 1.0
    assert 0.0 <= result["mc_accuracy"] <= 1.0
    assert result["mc_chance"] == pytest.approx(0.25)
    assert 0.0 <= result["cloze_exact_match_accuracy"] <= 1.0


def test_domain_probes_run_end_to_end_on_the_real_fixture(tiny_model, tokenizer):
    result = domain_probes.run(tiny_model, tokenizer, DEVICE)
    assert result["n_examples"] == 24
    assert 0.0 <= result["overall_accuracy"] <= 1.0
    assert "finance_term_accuracy" in result
    assert "proverb_accuracy" in result
    assert "advice_accuracy" in result


def test_generation_battery_returns_15_samples_with_diversity_metrics(tiny_model, tokenizer):
    result = generation.run(tiny_model, tokenizer, DEVICE, seed=0)
    assert len(result["samples"]) == 15
    for s in result["samples"]:
        assert 0.0 <= s["distinct_1"] <= 1.0
        assert 0.0 <= s["seq_rep_4"] <= 1.0
    assert 0.0 <= result["aggregate"]["distinct_1"] <= 1.0


def test_lambada_style_runs_on_the_real_val_books(tiny_model, tokenizer):
    result = benchmarks.run_lambada_style(
        tiny_model, tokenizer, "data/clean/val/books", DEVICE, max_examples=10, seed=0
    )
    assert result["n_examples"] == 10
    assert 0.0 <= result["last_word_accuracy"] <= 1.0


def test_hellaswag_runs_end_to_end(tiny_model, tokenizer):
    """Network-dependent (downloads/caches the real validation set) -- skip gracefully if
    offline rather than failing the whole suite. `tiny_model`'s max_seq_len=64 (vs the real
    S-tier's 512) means a few of HellaSwag's longer real-world endings legitimately get
    skipped by the too-long guard -- assert a lower bound, not an exact count."""
    try:
        result = benchmarks.run_hellaswag(tiny_model, tokenizer, DEVICE, max_examples=5, seed=0)
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"hellaswag dataset unavailable: {e}")
    assert result["n_examples"] + result["n_skipped_too_long"] == 5
    assert result["n_examples"] >= 1
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["chance"] == pytest.approx(0.25)
