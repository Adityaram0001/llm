#!/usr/bin/env python
"""CLI entrypoint for phase 4+ training runs — one config, one run folder.

Usage:
    python scripts/train.py --config configs/train_s_baseline.yaml
    python scripts/train.py --config configs/train_s_baseline.yaml --device cpu  # portability smoke test
    python scripts/train.py --resume experiments/20260712_p4_s-baseline
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import yaml

from llmlab.train.config import TrainConfig
from llmlab.train.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to a configs/train_*.yaml")
    parser.add_argument("--resume", type=Path, help="an existing experiments/<run_id> to continue")
    parser.add_argument("--run-id", type=str, default=None, help="override the auto-generated run_id")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "mps", "cuda"],
        help="override the config's device (e.g. --device cpu for the portability smoke test)",
    )
    parser.add_argument(
        "--wandb-online", action="store_true",
        help="override the config's wandb_mode to 'online' for live dashboard monitoring "
        "(D-009 default is offline; use this on cloud runs — needs WANDB_API_KEY in .env, "
        "see docs/WANDB.md)",
    )
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        if not (run_dir / "config.yaml").exists():
            parser.error(f"{run_dir} has no config.yaml -- not a run folder")
        cfg = TrainConfig.from_yaml(run_dir / "config.yaml")
        if args.device:
            cfg.device = args.device
        if args.wandb_online:
            cfg.logging.wandb_mode = "online"
        resume = True
    else:
        if args.config is None:
            parser.error("--config is required unless --resume is given")
        cfg = TrainConfig.from_yaml(args.config)
        if args.device:
            cfg.device = args.device
        if args.wandb_online:
            cfg.logging.wandb_mode = "online"
        slug = args.config.stem.removeprefix("train_").replace("_", "-")
        run_id = args.run_id or f"{datetime.date.today():%Y%m%d}_p{cfg.phase}_{slug}"
        run_dir = ROOT / "experiments" / run_id
        if run_dir.exists():
            parser.error(
                f"{run_dir} already exists -- pass --resume to continue it, or --run-id for a "
                "fresh one. Run folders are never overwritten (CLAUDE.md)."
            )
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
        resume = False

    print(f"run: {run_dir.name}  device: {cfg.device or '(auto)'}  resume: {resume}")
    trainer = Trainer(cfg, run_dir)
    if resume:
        trainer.load_checkpoint(run_dir / "ckpt" / "latest.pt")
        print(f"resumed from step {trainer.step}")

    status = trainer.fit()
    print(f"\n{status}: step {trainer.step}/{cfg.max_steps}, best val_loss {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
