"""Shared log-likelihood scoring primitives — every probe in this package (dictionary, domain,
HellaSwag, LAMBADA-style) reduces to "how likely is this continuation given this prompt", so
that logic lives here once instead of once per probe.

**Why length-normalized log-likelihood, not raw sum:** a base model can't follow instructions,
so there's no "pick the letter A/B/C/D" prompt to give it. Instead we score each candidate
continuation's own log-likelihood under the model and pick the highest — but a raw summed
log-likelihood is always higher (less negative) for a SHORTER continuation almost by construction
(fewer negative terms to add), which would silently bias every MC probe toward short wrong
answers. Dividing by token count (mean log-likelihood per token) removes that bias. This is the
same convention GPT-3's paper uses for LAMBADA/HellaSwag-style multiple-choice.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT


def encode_prompt_continuation(
    tokenizer: Tokenizer, prompt_text: str, continuation_text: str
) -> tuple[list[int], list[int]]:
    """Split `prompt_text + continuation_text`'s encoding at the prompt boundary, rather than
    encoding each half separately — BPE merges span whitespace at a boundary (e.g. "cat" + "s"
    might merge differently than "cats" encoded whole), so encoding the two halves independently
    can silently produce a different token sequence than the model would actually see reading the
    full text left to right. Encoding the whole string once and slicing is the standard
    lm-eval-harness-style trick for this."""
    prompt_ids = tokenizer.encode(prompt_text).ids
    full_ids = tokenizer.encode(prompt_text + continuation_text).ids
    return prompt_ids, full_ids[len(prompt_ids) :]


@torch.no_grad()
def score_continuation(
    model: GPT, prompt_ids: list[int], continuation_ids: list[int], device: torch.device
) -> tuple[float, float, int]:
    """Teacher-forced score of `continuation_ids` given `prompt_ids` as context.

    Returns `(sum_logprob, mean_logprob, n_tokens)`. Reuses `GPT.forward`'s own
    `ignore_index=-1` cross-entropy (the same convention the trainer uses) by masking every
    target position except the continuation's — so `loss` returned is already exactly the mean
    NLL over the continuation tokens, no separate softmax/gather code needed.

    Truncates from the LEFT of the prompt if `prompt+continuation` would exceed the model's
    `max_seq_len` (matches `generate()`'s own truncation convention) — the continuation itself
    is never truncated, since that's the thing being scored.
    """
    max_seq_len = model.cfg.max_seq_len
    full = prompt_ids + continuation_ids
    n_cont = len(continuation_ids)
    if len(full) > max_seq_len:
        full = full[-max_seq_len:]  # drop the earliest prompt tokens first
    n_prompt = len(full) - n_cont
    assert n_prompt >= 1, "prompt must have at least one token of context to score a continuation"

    idx = torch.tensor([full], dtype=torch.long, device=device)
    inputs = idx[:, :-1]
    targets = idx[:, 1:].clone()
    targets[:, : n_prompt - 1] = -1  # mask every target except the continuation's own tokens

    _, loss = model(inputs, targets)
    mean_logprob = -float(loss)
    return mean_logprob * n_cont, mean_logprob, n_cont


@torch.no_grad()
def mc_by_loglik(
    model: GPT, prompt_ids: list[int], choice_ids: list[list[int]], device: torch.device
) -> tuple[int, list[float]]:
    """Score each candidate continuation in `choice_ids` and return `(argmax_index, scores)`,
    `scores` = length-normalized mean log-likelihood per choice (see module docstring)."""
    scores = [score_continuation(model, prompt_ids, c, device)[1] for c in choice_ids]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


@torch.no_grad()
def greedy_continuation(
    model: GPT, prompt_ids: list[int], n_tokens: int, device: torch.device
) -> list[int]:
    """Deterministic (argmax) decode of `n_tokens` — `top_k=1` collapses `generate()`'s sampling
    to picking the single highest-probability token every step, reusing the tested generation
    path rather than a second decode implementation."""
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_tokens, top_k=1, use_cache=True)
    return out[0, len(prompt_ids) :].tolist()
