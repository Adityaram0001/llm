"""Stage a size-trimmed copy of experiments/ for R2 archival.

Runs on the pod (needs torch to strip optimizer state from checkpoints). For every run folder
under --experiments-dir, copies config.yaml/metrics.jsonl/notes.md/samples/ as-is (tiny) and
ckpt/best.pt with its optimizer_state_dict(s) stripped (model weights only — ablation runs are
reproducible from config+seed, so the archive doesn't need to be resumable). Runs named in
--fork-points are the exception: they've had real `--resume` runs fork off them, so both
ckpt/latest.pt and ckpt/best.pt are copied in full (optimizer state intact).

Usage: python archive_checkpoints.py --experiments-dir experiments --staging-dir _r2_staging \
    --fork-points wave_d_constant
"""

import argparse
import shutil
from pathlib import Path

import torch


def strip_optimizer(src: Path, dst: Path) -> tuple[int, int]:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt.pop("optimizer_state_dict", None)
    ckpt.pop("optimizer_state_dicts", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return src.stat().st_size, dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="experiments")
    ap.add_argument("--staging-dir", default="_r2_staging/experiments")
    ap.add_argument("--fork-points", default="", help="comma-separated run_id substrings kept full")
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir)
    staging = Path(args.staging_dir)
    fork_points = {s.strip() for s in args.fork_points.split(",") if s.strip()}

    total_orig = total_slim = 0
    rows = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "ckpt").exists():
            continue
        run_id = run_dir.name
        out_dir = staging / run_id
        for name in ("config.yaml", "metrics.jsonl", "notes.md"):
            f = run_dir / name
            if f.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out_dir / name)
        samples_dir = run_dir / "samples"
        if samples_dir.exists():
            shutil.copytree(samples_dir, out_dir / "samples", dirs_exist_ok=True)

        is_fork_point = run_id in fork_points
        for ckpt_name in ("best.pt", "latest.pt"):
            src = run_dir / "ckpt" / ckpt_name
            if not src.exists():
                continue
            if ckpt_name == "latest.pt" and not is_fork_point:
                continue  # archive only best.pt for ordinary runs
            dst = out_dir / "ckpt" / ckpt_name
            if is_fork_point:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                orig = slim = src.stat().st_size
            else:
                orig, slim = strip_optimizer(src, dst)
            total_orig += orig
            total_slim += slim
            rows.append((run_id, ckpt_name, orig, slim, is_fork_point))

    print(f"{'run_id':45s} {'file':10s} {'orig MB':>9s} {'slim MB':>9s}  fork_point")
    for run_id, name, orig, slim, fp in rows:
        print(f"{run_id:45s} {name:10s} {orig/1e6:9.1f} {slim/1e6:9.1f}  {fp}")
    print(f"\nTotal: {total_orig/1e6:.1f} MB -> {total_slim/1e6:.1f} MB staged at {staging}")


if __name__ == "__main__":
    main()
