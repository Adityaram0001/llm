"""Length-extrapolation probe (phase 5, Wave B): load a trained checkpoint, evaluate val loss
at sequence lengths beyond the training seq_len (e.g. train@512, eval@1024/2048). Only
pos_encoding in {rope, alibi, none} can run past max_seq_len (RW-5); learned/sinusoidal raise
ValueError by design (their positional table is physically sized to max_seq_len) -- this script
reports that as an expected failure rather than a crash.

Usage:
    python scripts/eval_extrapolation.py --run experiments/20260711_p4_s-baseline --seq-lens 512 1024 2048
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from llmlab.data.loader import MixedSourceLoader, Source
from llmlab.model import GPT, ModelConfig
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="experiments/<run_id>")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.run / "config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else get_device()
    model_cfg = ModelConfig.from_yaml(str(ROOT / train_cfg["model_config"]))
    model = GPT(model_cfg).to(device)
    ckpt = torch.load(args.run / "ckpt" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_sources = [
        Source(
            name=s["name"],
            bin_path=ROOT / s["path"],
            weight=s.get("weight", 1.0),
            respect_doc_boundaries=s.get("respect_doc_boundaries", False),
            docstarts_path=(ROOT / s["docstarts_path"]) if s.get("docstarts_path") else None,
        )
        for s in train_cfg["val_sources"]
    ]

    print(f"run={args.run.name}  pos_encoding={model_cfg.pos_encoding}  "
          f"trained max_seq_len={model_cfg.max_seq_len}  device={device}")
    for seq_len in args.seq_lens:
        loader = MixedSourceLoader(val_sources, seq_len, seed=train_cfg["seed"] + 1)
        batches = loader.fixed_eval_batches(args.eval_batches, args.eval_batch_size, device)
        try:
            with torch.no_grad():
                losses = [model(x, y)[1].item() for x, y in batches]
            val_loss = sum(losses) / len(losses)
            print(f"  seq_len={seq_len:5d}  val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f}")
        except ValueError as e:
            print(f"  seq_len={seq_len:5d}  FAILED (expected for bounded encodings): {e}")


if __name__ == "__main__":
    main()
