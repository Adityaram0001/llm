"""Generation battery — 15 fixed prompts sampled at a fixed temperature/top-p (frozen this
phase per the spec's decision point, D-046), saved side-by-side across checkpoints so a human
can eyeball how outputs change with training. Also reports distinct-n and a self-repetition
rate, since "reads as fluent English" and "isn't just repeating itself" are different failure
modes a loss number alone can't distinguish.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 80

PROMPTS = [
    # story openers
    "Once upon a time",
    "It was a dark and stormy night",
    "In a small village at the edge of the forest",
    "The old man had lived alone for many years",
    # "Define X:" — dictionary-format probes
    "ephemeral (adjective):",
    "prudence (noun):",
    "avarice (noun):",
    "diligence (noun):",
    # book-style prose (philosophical/reflective register, matches the corpus)
    "Of all the virtues a man may possess, none is more essential than",
    "Philosophy teaches us that true happiness consists in",
    "It is the nature of the human mind to seek",
    "Reason, rightly employed, is the surest guide to",
    # finance/wisdom prompts (RW-4 domain steer)
    "The first rule of saving money is",
    "A wise investor always considers",
    "The surest path to financial independence is",
]

assert len(PROMPTS) == 15


def _distinct_n(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def _seq_rep_4(tokens: list[int]) -> float:
    """Fraction of 4-grams that already occurred earlier in the same generation (Welleck et al.
    '20's `seq-rep-n`) -- a direct "is it stuck in a loop" signal, complementary to distinct-n
    (which is corpus-level and can look fine even with one long internal repeat)."""
    if len(tokens) < 4:
        return 0.0
    ngrams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    seen: set[tuple[int, ...]] = set()
    n_repeated = 0
    for g in ngrams:
        if g in seen:
            n_repeated += 1
        seen.add(g)
    return n_repeated / len(ngrams)


@torch.no_grad()
def run(model: GPT, tokenizer: Tokenizer, device: torch.device, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    samples = []
    all_gen_tokens: list[int] = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
        )
        gen_ids = out[0, len(prompt_ids) :].tolist()
        all_gen_tokens.extend(gen_ids)
        samples.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(gen_ids),
                "distinct_1": _distinct_n(gen_ids, 1),
                "distinct_2": _distinct_n(gen_ids, 2),
                "seq_rep_4": _seq_rep_4(gen_ids),
            }
        )

    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "samples": samples,
        "aggregate": {
            "distinct_1": _distinct_n(all_gen_tokens, 1),
            "distinct_2": _distinct_n(all_gen_tokens, 2),
            "seq_rep_4": _seq_rep_4(all_gen_tokens),
        },
    }
