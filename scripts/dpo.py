#!/usr/bin/env python
"""CLI entrypoint for phase 8 Part C DPO training — one config, one run folder.

Usage:
    python scripts/dpo.py --config configs/dpo_s_dictionary.yaml
    python scripts/dpo.py --config configs/dpo_s_dictionary.yaml --device cpu   # smoke test
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import yaml

from llmlab.train.dpo_config import DPOConfig
from llmlab.train.dpo_trainer import DPOTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="path to a configs/dpo_*.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="override the auto-generated run_id")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--wandb-online", action="store_true", help="stream to wandb (see docs/WANDB.md)")
    args = parser.parse_args()

    cfg = DPOConfig.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    if args.wandb_online:
        cfg.logging.wandb_mode = "online"

    slug = args.config.stem.removeprefix("dpo_").replace("_", "-")
    run_id = args.run_id or f"{datetime.date.today():%Y%m%d}_p{cfg.phase}_dpo-{slug}"
    run_dir = ROOT / "experiments" / run_id
    if run_dir.exists():
        parser.error(f"{run_dir} already exists -- pass --run-id for a fresh one (never overwritten).")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))

    print(f"run: {run_dir.name}  device: {cfg.device or '(auto)'}")
    trainer = DPOTrainer(cfg, run_dir)
    status = trainer.fit()
    print(f"\n{status}: {trainer.step}/{trainer.total_steps} steps, best DPO val_loss {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
