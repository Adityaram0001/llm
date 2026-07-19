#!/usr/bin/env python
"""Phase 8 Part C: before/after evaluation of a DPO model vs its SFT starting point (and base).

Three things, written to a markdown table + JSON in the DPO run folder:

  1. **Preference-pair reward accuracy** (the DPO-specific number): on the HELD-OUT DPO val
     triples, what fraction of the time does a policy's *implicit reward* rank chosen above
     rejected? Computed identically for two policies against the SAME frozen reference (the SFT
     model): "sft_as_policy" (does the pre-DPO model already prefer chosen without any DPO
     training?) and "dpo_as_policy" (does training raise that rate?) — the gap IS what DPO bought.
     Uses the exact same `dpo.sequence_logprobs`/`dpo_loss` primitives training used.
  2. **Instruction-following + dictionary retention**, base vs SFT vs DPO — same batteries as
     `eval_sft.py` Part A, extended to a third column so the whole A->B->C chain is comparable.
  3. **Catastrophic forgetting**, read from each run's own metrics.jsonl (not recomputed).

Usage:
    python scripts/eval_dpo.py --dpo-run experiments/20260719_p8_dpo-s-dictionary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from llmlab.data.chat_format import EOT, encode_prompt
from llmlab.data.dpo_loader import DPODataset
from llmlab.data.sft_loader import load_jsonl
from llmlab.eval import dictionary_probes
from llmlab.model import GPT, ModelConfig
from llmlab.train.dpo import dpo_loss, sequence_logprobs
from llmlab.train.sft_infer import load_finetuned
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


@torch.no_grad()
def reward_accuracy(
    policy: GPT, reference: GPT, val_ds: DPODataset, device: torch.device, batch_size: int = 16
) -> dict:
    """Mean reward_accuracy/margin of `policy` vs the frozen `reference`, over the held-out DPO
    val set — the same computation `DPOTrainer._val_metrics` runs live during training, exposed
    here so it can be re-run against ANY policy for comparison.

    NOTE: this is only meaningful when `policy is not reference`. When they ARE the same model
    (e.g. asking "does the pre-DPO SFT model already prefer chosen?"), every log-ratio is
    identically 0, so `reward_chosen == reward_rejected` for every example and the strict `>` in
    `reward_accuracy` reports a degenerate 0.0% — NOT evidence the model dispreferred chosen, just
    a tie. Use `raw_logprob_preference` below for that question instead."""
    accs, margins = [], []
    for x_c, y_c, x_r, y_r in val_ds.eval_batches(batch_size, device):
        pc, pr = sequence_logprobs(policy, x_c, y_c), sequence_logprobs(policy, x_r, y_r)
        rc, rr = sequence_logprobs(reference, x_c, y_c), sequence_logprobs(reference, x_r, y_r)
        _, m = dpo_loss(pc, pr, rc, rr, beta=0.1)
        accs.append(m["reward_accuracy"])
        margins.append(m["reward_margin"])
    return {"reward_accuracy": sum(accs) / len(accs), "reward_margin": sum(margins) / len(margins)}


@torch.no_grad()
def raw_logprob_preference(model: GPT, val_ds: DPODataset, device: torch.device, batch_size: int = 16) -> dict:
    """Does `model`, on its own (no reference needed), already assign higher sequence log-prob to
    chosen than rejected? The reference-free preference check — meaningful for ANY single model,
    including the pre-DPO SFT model, unlike `reward_accuracy` above."""
    accs, gaps = [], []
    for x_c, y_c, x_r, y_r in val_ds.eval_batches(batch_size, device):
        lp_c, lp_r = sequence_logprobs(model, x_c, y_c), sequence_logprobs(model, x_r, y_r)
        accs.append(float((lp_c > lp_r).float().mean()))
        gaps.append(float((lp_c - lp_r).mean()))
    return {"accuracy": sum(accs) / len(accs), "mean_logprob_gap": sum(gaps) / len(gaps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo-run", type=Path, required=True, help="experiments/<dpo run folder>")
    parser.add_argument("--n-instructions", type=int, default=100)
    parser.add_argument("--n-probes", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    dpo_cfg = yaml.safe_load((args.dpo_run / "config.yaml").read_text())
    sft_run_dir = ROOT / dpo_cfg["sft_run"]
    sft_cfg = yaml.safe_load((sft_run_dir / "config.yaml").read_text())
    model_config = dpo_cfg["model_config"]
    tokenizer = Tokenizer.from_file(str(ROOT / dpo_cfg["tokenizer_dir"] / "tokenizer.json"))

    base = load_model(model_config, ROOT / sft_cfg["base_checkpoint"], device)
    sft, _, _ = load_finetuned(sft_run_dir, dpo_cfg["sft_ckpt_name"], device)
    dpo, _, _ = load_finetuned(args.dpo_run, "best.pt", device)

    # 1. preference reward accuracy: SFT-as-policy vs DPO-as-policy, both against the frozen SFT reference
    print("reward accuracy on held-out DPO val pairs ...")
    val_ds = DPODataset.from_jsonl(
        ROOT / dpo_cfg["val_file"], tokenizer, max_len=dpo_cfg["max_len"],
        supervise_eot=dpo_cfg["supervise_eot"],
    )
    sft_as_policy = reward_accuracy(sft, sft, val_ds, device)  # degenerate (see docstring) -- kept for the training-metric parity
    dpo_as_policy = reward_accuracy(dpo, sft, val_ds, device)
    sft_raw_pref = raw_logprob_preference(sft, val_ds, device)
    dpo_raw_pref = raw_logprob_preference(dpo, val_ds, device)

    # 2. instruction-following battery, base/sft/dpo
    val_rows = load_jsonl(ROOT / sft_cfg["val_file"])[: args.n_instructions]
    instructions = [r["instruction"] for r in val_rows]
    print(f"instruction battery: {len(instructions)} held-out instructions ...")
    base_instr = instruction_battery(base, tokenizer, instructions, device)
    sft_instr = instruction_battery(sft, tokenizer, instructions, device)
    dpo_instr = instruction_battery(dpo, tokenizer, instructions, device)

    # 3. dictionary knowledge retention (RW-6-safe metrics only)
    val_dict = ROOT / "data/clean/val/dictionary.jsonl"
    print(f"dictionary probes: {args.n_probes} examples ...")
    base_probe = dictionary_probes.run(base, tokenizer, val_dict, device, max_examples=args.n_probes)
    sft_probe = dictionary_probes.run(sft, tokenizer, val_dict, device, max_examples=args.n_probes)
    dpo_probe = dictionary_probes.run(dpo, tokenizer, val_dict, device, max_examples=args.n_probes)

    # 4. forgetting, base -> sft -> dpo (chained: dpo's own probe starts where sft's ended)
    sft_metrics = [json.loads(l) for l in (sft_run_dir / "metrics.jsonl").read_text().splitlines() if l]
    dpo_metrics = [json.loads(l) for l in (args.dpo_run / "metrics.jsonl").read_text().splitlines() if l]
    pt_sft_end = next((m["pretrain_val_ppl"] for m in reversed(sft_metrics) if "pretrain_val_ppl" in m), None)
    pt_dpo_start = next((m["pretrain_val_ppl"] for m in dpo_metrics if "pretrain_val_ppl" in m), None)
    pt_dpo_end = next((m["pretrain_val_ppl"] for m in reversed(dpo_metrics) if "pretrain_val_ppl" in m), None)

    results = {
        "dpo_run": args.dpo_run.name,
        "sft_run": dpo_cfg["sft_run"],
        "preference_reward_accuracy": {"sft_as_policy": sft_as_policy, "dpo_as_policy": dpo_as_policy},
        "raw_logprob_preference": {"sft": sft_raw_pref, "dpo": dpo_raw_pref},
        "instruction_following": {"base": base_instr, "sft": sft_instr, "dpo": dpo_instr},
        "dictionary_retention": {
            "base": {k: base_probe[k] for k in ("mc_accuracy", "mc_chance", "cloze_exact_match_accuracy")},
            "sft": {k: sft_probe[k] for k in ("mc_accuracy", "mc_chance", "cloze_exact_match_accuracy")},
            "dpo": {k: dpo_probe[k] for k in ("mc_accuracy", "mc_chance", "cloze_exact_match_accuracy")},
            "note": "definition_completion_ppl omitted pending RW-6 (corrupted-text bug).",
        },
        "forgetting": {
            "pretrain_val_ppl_sft_end": pt_sft_end,
            "pretrain_val_ppl_dpo_start": pt_dpo_start,
            "pretrain_val_ppl_dpo_end": pt_dpo_end,
        },
    }
    out_path = (args.dpo_run / "eval_dpo.json").resolve()
    out_path.write_text(json.dumps(results, indent=2))

    def pct(x):
        return f"{100 * x:.1f}%"

    print("\n" + "=" * 70)
    print(f"PREFERENCE REWARD ACCURACY  ({args.dpo_run.name})")
    print("=" * 70)
    print(f"{'policy (vs frozen SFT reference)':<34}{'reward acc':>16}{'reward margin':>18}")
    print(f"{'SFT (pre-DPO)':<34}{pct(sft_as_policy['reward_accuracy']):>16}{sft_as_policy['reward_margin']:>18.4f}  <- degenerate tie, see note")
    print(f"{'DPO (post-training)':<34}{pct(dpo_as_policy['reward_accuracy']):>16}{dpo_as_policy['reward_margin']:>18.4f}")
    print(f"\n{'reference-free preference (does the model alone prefer chosen?)':<50}{'accuracy':>14}{'mean logp gap':>16}")
    print(f"{'SFT (pre-DPO)':<50}{pct(sft_raw_pref['accuracy']):>14}{sft_raw_pref['mean_logprob_gap']:>16.3f}")
    print(f"{'DPO (post-training)':<50}{pct(dpo_raw_pref['accuracy']):>14}{dpo_raw_pref['mean_logprob_gap']:>16.3f}")

    print("\n" + "=" * 70)
    print("BASE / SFT / DPO")
    print("=" * 70)
    print(f"{'metric':<34}{'base':>12}{'sft':>12}{'dpo':>12}")
    print(f"{'stop-rate':<34}{pct(base_instr['stop_rate']):>12}{pct(sft_instr['stop_rate']):>12}{pct(dpo_instr['stop_rate']):>12}")
    print(f"{'mean answer length (tok)':<34}{base_instr['mean_answer_len']:>12.1f}{sft_instr['mean_answer_len']:>12.1f}{dpo_instr['mean_answer_len']:>12.1f}")
    print(f"{'dict MC accuracy':<34}{pct(base_probe['mc_accuracy']):>12}{pct(sft_probe['mc_accuracy']):>12}{pct(dpo_probe['mc_accuracy']):>12}")
    print(f"{'dict cloze exact-match':<34}{pct(base_probe['cloze_exact_match_accuracy']):>12}{pct(sft_probe['cloze_exact_match_accuracy']):>12}{pct(dpo_probe['cloze_exact_match_accuracy']):>12}")
    print(f"{'pretrain val ppl':<34}{'-':>12}{pt_sft_end:>12.2f}{pt_dpo_end:>12.2f}")
    print("=" * 70)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
