#!/usr/bin/env python
"""Phase 8 Part A: before/after evaluation of an SFT model vs its pretrained base.

Measures three things and writes a markdown table + JSON into the SFT run folder:

  1. **Instruction-following battery** (the phase-8-specific eval): on held-out SFT val
     instructions, does the model *answer and stop* or *continue like a document?* Quantified as
     `stop_rate` (fraction that emit `<|endoftext|>` within the budget) and `mean_answer_len`
     (tokens before that stop). A base LM has no stop token in this protocol, so it just runs to
     the budget — the gap is the behavioral change SFT buys.

  2. **Dictionary knowledge retention** (P6 probes): MC-by-loglik accuracy and cloze exact-match,
     base vs SFT. These are the RW-6-*safe* probes; `definition_completion_ppl` is deliberately
     omitted (RW-6: it is computed on silently corrupted text until that bug is fixed).

  3. **Catastrophic forgetting**: pretrain-val perplexity is read from the SFT run's metrics.jsonl
     (logged live during training), not recomputed here.

Usage:
    python scripts/eval_sft.py --sft-run experiments/20260719_p8_sft-s-dictionary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from llmlab.data.chat_format import EOT, encode_prompt
from llmlab.data.sft_loader import load_jsonl
from llmlab.eval import dictionary_probes
from llmlab.model import GPT, ModelConfig
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]


def load_model(model_config: str, ckpt_path: Path, device: torch.device) -> GPT:
    cfg = ModelConfig.from_yaml(str(ROOT / model_config))
    model = GPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def instruction_battery(
    model: GPT, tokenizer: Tokenizer, instructions: list[str], device: torch.device, budget: int = 64
) -> dict:
    """For each instruction, feed the chat prompt and greedy-decode up to `budget` tokens. Report
    how often the model stops (emits EOT) and the mean length of what it produced before stopping."""
    eot_id = tokenizer.token_to_id(EOT)
    stops, lengths = 0, []
    for instr in instructions:
        ids = encode_prompt(tokenizer, instr)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=budget, top_k=1, use_cache=True)
        gen = out[0].tolist()[len(ids):]
        if eot_id in gen:
            stops += 1
            lengths.append(gen.index(eot_id))
        else:
            lengths.append(budget)
    return {
        "n": len(instructions),
        "stop_rate": stops / len(instructions),
        "mean_answer_len": sum(lengths) / len(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", type=Path, required=True, help="experiments/<sft run folder>")
    parser.add_argument("--n-instructions", type=int, default=100)
    parser.add_argument("--n-probes", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    sft_cfg = yaml.safe_load((args.sft_run / "config.yaml").read_text())
    model_config = sft_cfg["model_config"]
    tokenizer = Tokenizer.from_file(str(ROOT / sft_cfg["tokenizer_dir"] / "tokenizer.json"))

    base = load_model(model_config, ROOT / sft_cfg["base_checkpoint"], device)
    sft = load_model(model_config, args.sft_run / "ckpt" / "best.pt", device)

    # 1. instruction-following battery on held-out SFT val instructions
    val_rows = load_jsonl(ROOT / sft_cfg["val_file"])[: args.n_instructions]
    instructions = [r["instruction"] for r in val_rows]
    print(f"instruction battery: {len(instructions)} held-out instructions ...")
    base_instr = instruction_battery(base, tokenizer, instructions, device)
    sft_instr = instruction_battery(sft, tokenizer, instructions, device)

    # 2. dictionary knowledge retention (RW-6-safe metrics only)
    val_dict = ROOT / "data/clean/val/dictionary.jsonl"
    print(f"dictionary probes: {args.n_probes} examples ...")
    base_probe = dictionary_probes.run(base, tokenizer, val_dict, device, max_examples=args.n_probes)
    sft_probe = dictionary_probes.run(sft, tokenizer, val_dict, device, max_examples=args.n_probes)

    # 3. forgetting from the training log
    metrics = [json.loads(l) for l in (args.sft_run / "metrics.jsonl").read_text().splitlines() if l]
    pt0 = next((m["pretrain_val_ppl"] for m in metrics if "pretrain_val_ppl" in m), None)
    ptN = next((m["pretrain_val_ppl"] for m in reversed(metrics) if "pretrain_val_ppl" in m), None)

    results = {
        "sft_run": args.sft_run.name,
        "base_checkpoint": sft_cfg["base_checkpoint"],
        "instruction_following": {"base": base_instr, "sft": sft_instr},
        "dictionary_retention": {
            "base": {k: base_probe[k] for k in ("mc_accuracy", "mc_chance", "cloze_exact_match_accuracy")},
            "sft": {k: sft_probe[k] for k in ("mc_accuracy", "mc_chance", "cloze_exact_match_accuracy")},
            "note": "definition_completion_ppl omitted pending RW-6 (corrupted-text bug).",
        },
        "forgetting": {"pretrain_val_ppl_start": pt0, "pretrain_val_ppl_end": ptN},
    }
    out_path = (args.sft_run / "eval_sft.json").resolve()
    out_path.write_text(json.dumps(results, indent=2))

    def pct(x):
        return f"{100 * x:.1f}%"

    print("\n" + "=" * 64)
    print(f"BEFORE / AFTER SFT  ({args.sft_run.name})")
    print("=" * 64)
    print(f"{'metric':<34}{'base':>14}{'sft':>14}")
    print(f"{'stop-rate (answers & stops)':<34}{pct(base_instr['stop_rate']):>14}{pct(sft_instr['stop_rate']):>14}")
    print(f"{'mean answer length (tokens)':<34}{base_instr['mean_answer_len']:>14.1f}{sft_instr['mean_answer_len']:>14.1f}")
    print(f"{'dict MC accuracy (chance 25%)':<34}{pct(base_probe['mc_accuracy']):>14}{pct(sft_probe['mc_accuracy']):>14}")
    print(f"{'dict cloze exact-match':<34}{pct(base_probe['cloze_exact_match_accuracy']):>14}{pct(sft_probe['cloze_exact_match_accuracy']):>14}")
    print(f"{'pretrain val ppl (forgetting)':<34}{pt0:>14.2f}{ptN:>14.2f}")
    print("=" * 64)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
