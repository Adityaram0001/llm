#!/usr/bin/env python
"""Phase 8 Part B: compare full fine-tuning vs LoRA across quality, trainable params, optimizer
memory, checkpoint size, and throughput. Reads each run's config/metrics/checkpoint and (unless
--no-instr) runs the instruction-following battery on each. Writes a markdown table to stdout and
a JSON blob to docs/results/finetune_partB.json.

Usage:
    python scripts/compare_finetune.py \
        --full experiments/20260719_p8_sft-s-dictionary \
        --lora experiments/20260719_p8_sft-s-dictionary-lora-r8-attn \
               experiments/20260719_p8_sft-s-dictionary-lora-r32-attn \
               experiments/20260719_p8_sft-s-dictionary-lora-r8-attnffn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from llmlab.data.chat_format import EOT, encode_prompt
from llmlab.data.sft_loader import load_jsonl
from llmlab.model import GPT, ModelConfig
from llmlab.train.lora import apply_lora, lora_parameters, optimizer_state_bytes
from llmlab.train.sft_infer import load_finetuned
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]


def trainable_count(cfg: dict) -> int:
    """Rebuild the model (+ LoRA, if configured) from the run config and count trainable params —
    the exact quantity AdamW keeps gradient + 2 moments for."""
    model = GPT(ModelConfig.from_yaml(str(ROOT / cfg["model_config"])))
    if cfg.get("lora"):
        lc = cfg["lora"]
        apply_lora(model, lc["targets"], r=lc["r"], alpha=lc["alpha"], dropout=lc.get("dropout", 0.0))
        return sum(p.numel() for p in lora_parameters(model))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def instr_stop_rate(run_dir: Path, instructions: list[str], device: torch.device) -> float:
    model, tokenizer, _ = load_finetuned(run_dir, "best.pt", device)
    eot_id = tokenizer.token_to_id(EOT)
    stops = 0
    for instr in instructions:
        ids = encode_prompt(tokenizer, instr)
        out = model.generate(
            torch.tensor([ids], device=device), max_new_tokens=64, top_k=1, use_cache=True
        )
        if eot_id in out[0].tolist()[len(ids):]:
            stops += 1
    return stops / len(instructions)


def run_summary(run_dir: Path, instructions: list[str] | None, device: torch.device) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l]
    best_val = min(m["val_loss"] for m in metrics if "val_loss" in m)
    pt = [m["pretrain_val_ppl"] for m in metrics if "pretrain_val_ppl" in m]
    toks = [m["tok_s"] for m in metrics if m.get("tok_s")]
    n_train = trainable_count(cfg)
    best_pt = run_dir / "ckpt" / "best.pt"
    return {
        "run": run_dir.name,
        "mode": f"LoRA r{cfg['lora']['r']} {cfg['lora']['targets']}" if cfg.get("lora") else "full FT",
        "lr": cfg["lr"],
        "trainable_params": n_train,
        "trainable_pct": 100 * n_train / (trainable_count({**cfg, "lora": None})),
        "opt_state_mb": optimizer_state_bytes(n_train) / 1e6,
        "ckpt_mb": best_pt.stat().st_size / 1e6 if best_pt.exists() else None,
        "mean_tok_s": sum(toks) / len(toks) if toks else None,
        "best_val_loss": best_val,
        "pretrain_ppl_start": pt[0] if pt else None,
        "pretrain_ppl_end": pt[-1] if pt else None,
        "stop_rate": instr_stop_rate(run_dir, instructions, device) if instructions is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--lora", type=Path, nargs="+", required=True)
    parser.add_argument("--n-instructions", type=int, default=100)
    parser.add_argument("--no-instr", action="store_true", help="skip the instruction-following eval")
    args = parser.parse_args()

    device = get_device()
    instructions = None
    if not args.no_instr:
        full_cfg = yaml.safe_load((args.full / "config.yaml").read_text())
        rows = load_jsonl(ROOT / full_cfg["val_file"])[: args.n_instructions]
        instructions = [r["instruction"] for r in rows]

    summaries = [run_summary(d, instructions, device) for d in [args.full, *args.lora]]
    (ROOT / "docs/results/finetune_partB.json").write_text(json.dumps(summaries, indent=2))

    cols = [
        ("mode", "{:<18}", "{:<18}"),
        ("trainable_params", "{:>12}", "{:>12,}"),
        ("trainable_pct", "{:>8}", "{:>7.2f}%"),
        ("opt_state_mb", "{:>10}", "{:>9.2f}M"),
        ("ckpt_mb", "{:>9}", "{:>8.2f}M"),
        ("mean_tok_s", "{:>9}", "{:>9.0f}"),
        ("best_val_loss", "{:>8}", "{:>8.3f}"),
        ("stop_rate", "{:>7}", "{:>6.0%}"),
    ]
    header = "".join(h.format(name) for name, h, _ in cols)
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print("".join(fmt.format(s[name]) if s[name] is not None else f"{'-':>6}" for name, _, fmt in cols))
    print("\nwrote docs/results/finetune_partB.json")


if __name__ == "__main__":
    main()
