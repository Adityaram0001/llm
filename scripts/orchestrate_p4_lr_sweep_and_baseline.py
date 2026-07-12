#!/usr/bin/env python
"""Unattended overnight pipeline (2026-07-11 session, user's explicit request): run the 3
p4_s_lr_sweep configs sequentially, pick the winning lr by lowest logged val_loss (a run that
ever produces a non-finite train_loss is disqualified -- that's the "watch divergence happen
on purpose" lr_hi candidate doing its job), write the winner into a fresh baseline config, then
launch the full p4_s_baseline run with it. Zero interaction required: never prompts, logs
everything, and each run gets its own notes.md so the lab record is complete by morning.

**This is a provisional automation, not a ratified decision**: the winning lr replaces
D-021's default (1e-3) for THIS baseline run only; `configs/train_s_baseline.yaml` on disk is
left untouched. The next session should review the sweep + the chosen lr before treating it as
settled (see the generated `configs/train_s_baseline_auto.yaml` and each run's notes.md).

Usage (normally launched detached via nohup+caffeinate, see the session's PROGRESS.md note):
    python scripts/orchestrate_p4_lr_sweep_and_baseline.py
"""

from __future__ import annotations

import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
DATE = datetime.date.today().strftime("%Y%m%d")

SWEEP_CONFIGS = [
    ROOT / "configs" / "train_s_lr_sweep_lo.yaml",
    ROOT / "configs" / "train_s_lr_sweep_mid.yaml",
    ROOT / "configs" / "train_s_lr_sweep_hi.yaml",
]


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_train(config_path: Path, run_id: str) -> int:
    log(f"launching {run_id} ({config_path.name})")
    result = subprocess.run(
        [str(VENV_PYTHON), str(ROOT / "scripts" / "train.py"), "--config", str(config_path), "--run-id", run_id],
        cwd=ROOT,
    )
    log(f"{run_id} exited with code {result.returncode}")
    return result.returncode


def best_val_loss_for_run(run_id: str) -> float | None:
    """Lowest logged val_loss for a run; None if it never logged one, or if `train_loss` went
    non-finite anywhere (NaN/Inf) -- that disqualifies the whole run as diverged."""
    metrics_path = ROOT / "experiments" / run_id / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    val_losses = []
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if not math.isfinite(rec["train_loss"]):
            return None
        if "val_loss" in rec:
            if not math.isfinite(rec["val_loss"]):
                return None
            val_losses.append(rec["val_loss"])
    return min(val_losses) if val_losses else None


def write_notes(run_dir: Path, hypothesis: str, observation: str, conclusion: str) -> None:
    (run_dir / "notes.md").write_text(
        f"# {run_dir.name}\n\n**Hypothesis:** {hypothesis}\n\n**Observation:** {observation}\n\n"
        f"**Conclusion:** {conclusion}\n"
    )


def main() -> None:
    sweep_results: dict[str, tuple[float, float | None]] = {}

    for config_path in SWEEP_CONFIGS:
        cfg = yaml.safe_load(config_path.read_text())
        lr = cfg["optim"]["lr"]
        slug = config_path.stem.removeprefix("train_").replace("_", "-")
        run_id = f"{DATE}_p4_{slug}"
        run_dir = ROOT / "experiments" / run_id

        if run_dir.exists():
            log(f"{run_id} already exists -- reusing its metrics instead of re-running")
        else:
            run_train(config_path, run_id)

        val_loss = best_val_loss_for_run(run_id)
        sweep_results[run_id] = (lr, val_loss)
        status = f"best_val_loss={val_loss:.4f}" if val_loss is not None else "DIVERGED or no metrics"
        write_notes(
            run_dir,
            hypothesis=(
                f"lr={lr:.1e} vs the D-021 baseline lr (1e-3, tested as lr_sweep_mid) -- part of "
                "phase 4's lr-sweep lesson (watch under/over-shooting happen on purpose)."
            ),
            observation=f"300 steps, ~19.7M tokens. {status}. Full curve in metrics.jsonl.",
            conclusion=(
                "Written automatically by scripts/orchestrate_p4_lr_sweep_and_baseline.py as "
                "part of the 2026-07-11 overnight sweep-then-baseline pipeline; see that run's "
                "sibling runs for the 3-way comparison and docs/DECISIONS.md for the outcome."
            ),
        )

    log("\n=== lr sweep results ===")
    for run_id, (lr, val_loss) in sweep_results.items():
        shown = f"{val_loss:.4f}" if val_loss is not None else "DIVERGED/MISSING"
        log(f"  {run_id}: lr={lr:.1e}  best_val_loss={shown}")

    valid = {rid: (lr, vl) for rid, (lr, vl) in sweep_results.items() if vl is not None}
    if not valid:
        log("All lr-sweep runs diverged or produced no val_loss -- aborting before the baseline run.")
        sys.exit(1)

    winner_run_id, (winner_lr, winner_val_loss) = min(valid.items(), key=lambda item: item[1][1])
    log(f"winning lr: {winner_lr:.1e} (from {winner_run_id}, val_loss={winner_val_loss:.4f})")

    baseline_cfg_path = ROOT / "configs" / "train_s_baseline.yaml"
    baseline_cfg = yaml.safe_load(baseline_cfg_path.read_text())
    original_lr = baseline_cfg["optim"]["lr"]
    baseline_cfg["optim"]["lr"] = winner_lr
    auto_cfg_path = ROOT / "configs" / "train_s_baseline_auto.yaml"
    auto_cfg_path.write_text(yaml.dump(baseline_cfg, sort_keys=False))
    log(f"wrote {auto_cfg_path.name} (lr {original_lr:.1e} -> {winner_lr:.1e}); D-021's own "
        f"configs/train_s_baseline.yaml is left untouched")

    baseline_run_id = f"{DATE}_p4_s-baseline"
    if (ROOT / "experiments" / baseline_run_id).exists():
        baseline_run_id = f"{DATE}_p4_s-baseline-auto"
    rc = run_train(auto_cfg_path, baseline_run_id)

    baseline_val_loss = best_val_loss_for_run(baseline_run_id)
    write_notes(
        ROOT / "experiments" / baseline_run_id,
        hypothesis=(
            "THE S-tier reference run for phase 4/5 comparisons -- lr auto-selected overnight "
            f"from the sweep above ({winner_lr:.1e}, replacing D-021's default {original_lr:.1e} "
            "for this run only; needs next-session review, see this script's docstring)."
        ),
        observation=(
            f"1500 steps, ~98.3M tokens. "
            + (f"best_val_loss={baseline_val_loss:.4f}." if baseline_val_loss is not None else "DIVERGED or incomplete.")
        ),
        conclusion="Review val_loss curve in notebooks/05_compare_runs.ipynb; ratify or override the auto-picked lr in DECISIONS.md next session.",
    )
    log(f"pipeline complete, baseline exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
