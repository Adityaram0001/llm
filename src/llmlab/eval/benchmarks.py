"""Standard tiny-model-appropriate benchmarks, implemented by hand (not lm-eval-harness) per
the phase 6 spec — reimplementing the scoring logic IS the lesson.

**HellaSwag** (Zellers et al. '19): 4-way sentence-completion by log-likelihood, via the real
public validation set (`Rowan/hellaswag` on the HF Hub — the original `hellaswag` loading
script is deprecated by newer `datasets` versions, this is a maintained parquet mirror of the
same data). **Expect near-chance (25%) accuracy at this scale** — HellaSwag's distractors are
adversarially filtered (Zellers et al. specifically generated them to fool an earlier LM while
still being obviously wrong to a human), which requires broad world knowledge and commonsense
physics/social reasoning a ~10M-param model trained on ~18M tokens of 19th-century books and a
dictionary simply has no way to have absorbed. A near-chance score here isn't a bug to fix —
it's the correct, informative outcome, and worth contrasting with the dictionary probes'
real-above-chance accuracy (same MC-by-loglik mechanism, different task).

**LAMBADA-style last-word accuracy**: LAMBADA itself is a curated set of long narrative passages
where predicting the final word requires the FULL passage's context (a short excerpt isn't
enough) — building that curation is its own project. This is a homemade proxy using the same
"predict the final word" mechanic on our own held-out book val split (`data/clean/val/books/`),
via greedy decode + exact match, not the curated long-range-dependency version.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT

from .scoring import encode_prompt_continuation, greedy_continuation, mc_by_loglik

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@torch.no_grad()
def run_hellaswag(
    model: GPT, tokenizer: Tokenizer, device: torch.device, max_examples: int = 200, seed: int = 0
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:max_examples]

    correct = 0
    n_scored = 0
    n_skipped = 0
    for i in idxs:
        row = ds[i]
        ctx, endings, label = row["ctx"], row["endings"], int(row["label"])
        prompt_ids = tokenizer.encode(ctx).ids
        choice_ids = [encode_prompt_continuation(tokenizer, ctx, " " + e)[1] for e in endings]
        # HellaSwag's wikiHow-style contexts occasionally run long enough that, combined with a
        # long ending, `score_continuation`'s left-truncation to max_seq_len would leave zero
        # prompt tokens (only possible when an ending alone is >= max_seq_len; a non-issue at
        # the S-tier's 512, but real enough in HellaSwag's actual data to guard rather than let
        # one outlier crash a 10-minute run).
        if max(len(c) for c in choice_ids) >= model.cfg.max_seq_len:
            n_skipped += 1
            continue
        pred_idx, _ = mc_by_loglik(model, prompt_ids, choice_ids, device)
        correct += int(pred_idx == label)
        n_scored += 1

    return {
        "n_examples": n_scored,
        "n_skipped_too_long": n_skipped,
        "accuracy": correct / n_scored if n_scored else 0.0,
        "chance": 0.25,
    }


def _sentences_from_books(val_books_dir: str | Path, min_words: int = 8) -> list[str]:
    sentences = []
    for path in sorted(Path(val_books_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").replace("\n", " ")
        for s in _SENTENCE_SPLIT_RE.split(text):
            s = s.strip()
            if len(s.split()) >= min_words:
                sentences.append(s)
    return sentences


@torch.no_grad()
def run_lambada_style(
    model: GPT,
    tokenizer: Tokenizer,
    val_books_dir: str | Path,
    device: torch.device,
    max_examples: int = 150,
    seed: int = 0,
) -> dict:
    sentences = _sentences_from_books(val_books_dir)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    subset = sentences[:max_examples]

    correct = 0
    for sentence in subset:
        words = sentence.split()
        prompt_text = " ".join(words[:-1])
        cont_text = " " + words[-1]
        prompt_ids, cont_ids = encode_prompt_continuation(tokenizer, prompt_text, cont_text)
        pred_ids = greedy_continuation(model, prompt_ids, len(cont_ids), device)
        correct += int(pred_ids == cont_ids)

    return {"n_examples": len(subset), "last_word_accuracy": correct / len(subset)}
