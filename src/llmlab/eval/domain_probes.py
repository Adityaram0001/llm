"""Domain probes (RW-4): does the domain-mix ablation's finance/self-help/wisdom share actually
steer the model, beyond just costing general val_loss (Wave G, D-045)? Phase 7's data factory
(not built yet) was the spec's suggested source for these items; since it doesn't exist, this
phase hand-writes a small fixed set instead (`data/eval/domain_probes.json`, 24 items across 3
categories: finance-term definitions, proverb/maxim completion, sound-advice-vs-nonsense) —
same MC-by-loglik pattern as the dictionary probes, just a different fixed item set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmlab.model import GPT

from .scoring import encode_prompt_continuation, mc_by_loglik

DEFAULT_PROBES_PATH = Path(__file__).resolve().parents[3] / "data" / "eval" / "domain_probes.json"


@torch.no_grad()
def run(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    probes_path: str | Path = DEFAULT_PROBES_PATH,
) -> dict:
    items = json.loads(Path(probes_path).read_text(encoding="utf-8"))

    per_category: dict[str, list[int]] = {}
    for item in items:
        prompt_ids = tokenizer.encode(item["prompt"]).ids
        choice_ids = []
        for choice in item["choices"]:
            # leading space so the continuation tokenizes the way it would mid-sentence
            _, cont_ids = encode_prompt_continuation(tokenizer, item["prompt"], " " + choice)
            choice_ids.append(cont_ids)
        pred_idx, _ = mc_by_loglik(model, prompt_ids, choice_ids, device)
        correct = int(pred_idx == item["correct_idx"])
        per_category.setdefault(item["category"], []).append(correct)

    result = {
        "n_examples": len(items),
        "mc_chance": 0.25,
        "overall_accuracy": sum(sum(v) for v in per_category.values()) / len(items),
    }
    for category, correct_flags in per_category.items():
        result[f"{category}_accuracy"] = sum(correct_flags) / len(correct_flags)
    return result
