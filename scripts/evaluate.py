#!/usr/bin/env python
"""Phase 6 CLI entrypoint — score any checkpoint on the fixed eval battery.

Usage:
    python scripts/evaluate.py --ckpt experiments/<run>/ckpt/best.pt
    python scripts/evaluate.py --ckpt experiments/<run>/ckpt/best.pt --suite core --device cpu

Writes `eval_results.json` into the checkpoint's run folder (`<ckpt's parent's parent>/`).
The model config and tokenizer are read from that run folder's own `config.yaml` — the exact
resolved config the trainer dumped at run start (`scripts/train.py`) — so no separate
`--model-config`/`--tokenizer-dir` flags are needed for a normal `experiments/<run>/ckpt/*.pt`
checkpoint.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from llmlab.eval import run_core_suite
from llmlab.model import GPT, ModelConfig
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="path to a .pt checkpoint under experiments/<run>/ckpt/")
    parser.add_argument("--suite", type=str, default="core", choices=["core"], help="which eval battery to run")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=Path, default=None, help="override the output path (default: <run_dir>/eval_results.json)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-examples", type=str, default=None,
        help='JSON dict overriding per-probe example caps, e.g. \'{"hellaswag": 20}\' for a fast smoke test',
    )
    args = parser.parse_args()

    run_dir = args.ckpt.resolve().parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        parser.error(f"{config_path} not found -- expected a train run folder at {run_dir}")
    cfg = yaml.safe_load(config_path.read_text())

    device = torch.device(args.device) if args.device else get_device()
    model_cfg = ModelConfig.from_yaml(ROOT / cfg["model_config"])
    model = GPT(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(ROOT / cfg["tokenizer_dir"] / "tokenizer.json"))

    max_examples = json.loads(args.max_examples) if args.max_examples else None

    print(f"evaluating {args.ckpt} (step {ckpt['step']}, device {device}) ...")
    suite_result = run_core_suite(model, tokenizer, device, max_examples=max_examples, seed=args.seed)

    output = {
        "checkpoint": str(args.ckpt.resolve().relative_to(ROOT)),
        "step": ckpt["step"],
        "suite": args.suite,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **suite_result,
    }

    out_path = args.out or (run_dir / "eval_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}  (total wall time: {suite_result['total_wall_s']:.1f}s)")

    r = suite_result["results"]
    print(f"  books ppl:       {r['perplexity']['books']['ppl']:.3f}  bpb {r['perplexity']['books']['bits_per_byte']:.3f}")
    print(f"  dictionary ppl:  {r['perplexity']['dictionary']['ppl']:.3f}  bpb {r['perplexity']['dictionary']['bits_per_byte']:.3f}")
    print(f"  dict def-compl ppl: {r['dictionary_probes']['definition_completion_ppl']:.3f}  mc_acc {r['dictionary_probes']['mc_accuracy']:.3f} (chance {r['dictionary_probes']['mc_chance']:.2f})  cloze_acc {r['dictionary_probes']['cloze_exact_match_accuracy']:.3f}")
    print(f"  domain probes:   overall_acc {r['domain_probes']['overall_accuracy']:.3f} (chance {r['domain_probes']['mc_chance']:.2f})")
    print(f"  hellaswag:       acc {r['benchmarks']['hellaswag']['accuracy']:.3f} (chance {r['benchmarks']['hellaswag']['chance']:.2f})")
    print(f"  lambada-style:   last-word acc {r['benchmarks']['lambada_style']['last_word_accuracy']:.3f}")


if __name__ == "__main__":
    main()
