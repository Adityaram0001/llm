"""Dictionary probes — this project's "special sauce" (CLAUDE.md/phase6 spec): the corpus
includes the full GCIDE dictionary (D-006/D-012), so a base model trained on it should show some
measurable "knows what words mean" signal beyond generic prose modeling. Three angles on the
same `dictionary.jsonl` val entries (`{"word", "pos", "definitions"}`):

  (a) definition completion ppl  — teacher-forced ppl of the GOLD definition given the headword
  (b) multiple-choice define     — gold definition vs 3 shuffled wrong ones, picked by log-lik
  (c) cloze                      — given the definition, greedy-decode the headword (reversed)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT

from .scoring import encode_prompt_continuation, greedy_continuation, mc_by_loglik, score_continuation

N_DISTRACTORS = 3


def _load_entries(path: str | Path) -> list[dict]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _prompt(word: str, pos: str) -> str:
    return f"{word} ({pos}): " if pos else f"{word}: "


@torch.no_grad()
def run(
    model: GPT,
    tokenizer: Tokenizer,
    val_path: str | Path,
    device: torch.device,
    max_examples: int = 200,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    entries = [e for e in _load_entries(val_path) if e["definitions"]]
    rng.shuffle(entries)
    subset = entries[:max_examples]

    # (a) definition completion ppl — corpus-level (sum NLL / sum tokens), same convention as
    # perplexity.py, so this number is comparable across checkpoints the same way val ppl is.
    total_nll, total_tokens = 0.0, 0
    for e in subset:
        prompt_ids, cont_ids = encode_prompt_continuation(
            tokenizer, _prompt(e["word"], e["pos"]), e["definitions"][0]
        )
        sum_lp, _, n = score_continuation(model, prompt_ids, cont_ids, device)
        total_nll += -sum_lp
        total_tokens += n
    definition_completion_ppl = math.exp(total_nll / total_tokens)

    # (b) multiple-choice define: gold vs 3 distractor definitions from OTHER entries, chance=25%
    mc_correct = 0
    all_definitions = [e["definitions"][0] for e in entries]
    for e in subset:
        prompt_ids = tokenizer.encode(_prompt(e["word"], e["pos"])).ids
        distractor_pool = [d for d in all_definitions if d != e["definitions"][0]]
        distractors = rng.sample(distractor_pool, N_DISTRACTORS)
        choices_text = [e["definitions"][0]] + distractors
        order = list(range(len(choices_text)))
        rng.shuffle(order)
        correct_idx = order.index(0)
        choice_ids = [tokenizer.encode(choices_text[i]).ids for i in order]
        pred_idx, _ = mc_by_loglik(model, prompt_ids, choice_ids, device)
        mc_correct += int(pred_idx == correct_idx)
    mc_accuracy = mc_correct / len(subset)

    # (c) cloze: reverse the task — given the definition, greedy-decode the headword.
    cloze_correct = 0
    cloze_nll, cloze_tokens = 0.0, 0
    for e in subset:
        prompt_text = f"Definition: {e['definitions'][0]}\nThe word being defined is:"
        cont_text = f" {e['word']}"
        prompt_ids, cont_ids = encode_prompt_continuation(tokenizer, prompt_text, cont_text)
        sum_lp, _, n = score_continuation(model, prompt_ids, cont_ids, device)
        cloze_nll += -sum_lp
        cloze_tokens += n
        pred_ids = greedy_continuation(model, prompt_ids, len(cont_ids), device)
        cloze_correct += int(pred_ids == cont_ids)
    cloze_ppl = math.exp(cloze_nll / cloze_tokens)
    cloze_accuracy = cloze_correct / len(subset)

    return {
        "n_examples": len(subset),
        "definition_completion_ppl": definition_completion_ppl,
        "mc_accuracy": mc_accuracy,
        "mc_chance": 1.0 / (N_DISTRACTORS + 1),
        "cloze_ppl": cloze_ppl,
        "cloze_exact_match_accuracy": cloze_accuracy,
    }
