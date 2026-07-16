"""Orchestrates the phase-6 "core" eval suite: one call in, one JSON-safe dict out.
`scripts/evaluate.py` is the CLI wrapper around `run_core_suite`.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT

from . import benchmarks, dictionary_probes, domain_probes, generation, perplexity

ROOT = Path(__file__).resolve().parents[3]

BOOKS_VAL_BIN = ROOT / "data" / "tokenized" / "hf_bpe_16k" / "books_only_val.bin"
DICTIONARY_VAL_BIN = ROOT / "data" / "tokenized" / "hf_bpe_16k" / "dictionary_only_val.bin"
DICTIONARY_VAL_JSONL = ROOT / "data" / "clean" / "val" / "dictionary.jsonl"
VAL_BOOKS_DIR = ROOT / "data" / "clean" / "val" / "books"


def run_core_suite(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    max_examples: dict[str, int] | None = None,
    seed: int = 0,
) -> dict:
    """`max_examples` overrides the per-probe example cap, e.g. `{"hellaswag": 50}` for a fast
    smoke test — defaults are sized to keep the whole suite under the phase's 10-minute exit
    criterion on an S-tier model."""
    max_examples = max_examples or {}
    model.eval()
    t0 = time.time()
    timings: dict[str, float] = {}

    def _timed(name: str, fn):
        t = time.time()
        result = fn()
        timings[name] = round(time.time() - t, 2)
        return result

    results = {
        "perplexity": {
            "books": _timed(
                "perplexity_books",
                lambda: perplexity.evaluate_split(model, tokenizer, str(BOOKS_VAL_BIN), device),
            ),
            "dictionary": _timed(
                "perplexity_dictionary",
                lambda: perplexity.evaluate_split(model, tokenizer, str(DICTIONARY_VAL_BIN), device),
            ),
        },
        "dictionary_probes": _timed(
            "dictionary_probes",
            lambda: dictionary_probes.run(
                model,
                tokenizer,
                DICTIONARY_VAL_JSONL,
                device,
                max_examples=max_examples.get("dictionary_probes", 200),
                seed=seed,
            ),
        ),
        "domain_probes": _timed(
            "domain_probes", lambda: domain_probes.run(model, tokenizer, device)
        ),
        "generation": _timed(
            "generation", lambda: generation.run(model, tokenizer, device, seed=seed)
        ),
        "benchmarks": {
            "hellaswag": _timed(
                "hellaswag",
                lambda: benchmarks.run_hellaswag(
                    model, tokenizer, device, max_examples=max_examples.get("hellaswag", 200), seed=seed
                ),
            ),
            "lambada_style": _timed(
                "lambada_style",
                lambda: benchmarks.run_lambada_style(
                    model,
                    tokenizer,
                    VAL_BOOKS_DIR,
                    device,
                    max_examples=max_examples.get("lambada_style", 150),
                    seed=seed,
                ),
            ),
        },
    }

    return {
        "results": results,
        "timings_s": timings,
        "total_wall_s": round(time.time() - t0, 2),
    }
