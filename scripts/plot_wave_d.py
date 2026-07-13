#!/usr/bin/env python
"""Wave D figure: optimizers, schedules (+ the WSD multi-budget fork demo), the batch-size
study, and the grad-clip-off comparison -- four panels from one shared set of metrics.jsonl
files, saved to docs/results/wave_d_optimizers_schedules.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = 0.015  # D-035 seed spread


def curve(run_id: str, key: str = "val_loss") -> tuple[list[float], list[float]]:
    toks, vals = [], []
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get(key) is not None:
                toks.append(d["tokens_seen"] / 1e6)
                vals.append(d[key])
    return toks, vals


def final_val(run_id: str) -> float:
    t, v = curve(run_id)
    return v[-1]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_opt, ax_sched, ax_batch, ax_clip = axes.flat

    # -- (a) optimizers -------------------------------------------------------
    control_final = final_val("20260713_p5_s-wave-d-control")
    for label, run_id, color in [
        ("AdamW (control)", "20260713_p5_s-wave-d-control", "#4C6EF5"),
        ("Muon", "20260713_p5_s-wave-d-muon", "#F76707"),
        ("Lion", "20260713_p5_s-wave-d-lion", "#94D82D"),
    ]:
        t, v = curve(run_id)
        ax_opt.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_opt.axhspan(control_final - NOISE_FLOOR, control_final + NOISE_FLOOR, color="grey",
                   alpha=0.15, label=f"control ± noise floor ({NOISE_FLOOR})")
    ax_opt.set_title("Optimizers (same ~98.3M token budget)")
    ax_opt.set_xlabel("tokens seen (M)"); ax_opt.set_ylabel("val loss"); ax_opt.legend(fontsize=8)

    # -- (b) schedules + WSD multi-budget fork ---------------------------------
    for label, run_id, color in [
        ("cosine (control)", "20260713_p5_s-wave-d-control", "#4C6EF5"),
        ("WSD (decay last 20%)", "20260713_p5_s-wave-d-wsd", "#F76707"),
        ("constant (no decay)", "20260713_p5_s-wave-d-constant", "#94D82D"),
    ]:
        t, v = curve(run_id)
        ax_sched.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    for label, run_id, color in [
        ("+ short decay fork (→{:.3f})".format(final_val("20260713_p5_s-wave-d-wsd_branch_short")),
         "20260713_p5_s-wave-d-wsd_branch_short", "#AE3EC9"),
        ("+ long decay fork (→{:.3f})".format(final_val("20260713_p5_s-wave-d-wsd_branch_long")),
         "20260713_p5_s-wave-d-wsd_branch_long", "#E64980"),
    ]:
        t, v = curve(run_id)
        ax_sched.plot(t, v, label=label, color=color, lw=1.8, ls="--")
    ax_sched.axvline(1500 * 65536 / 1e6, color="grey", lw=1, ls=":",
                      label="fork point (step 1500)")
    ax_sched.set_title("Schedules + WSD multi-budget bonus\n(forks share wave_d_constant's step-1500 checkpoint)")
    ax_sched.set_xlabel("tokens seen (M)"); ax_sched.set_ylabel("val loss"); ax_sched.legend(fontsize=7)

    # -- (c) batch-size study ---------------------------------------------------
    batch_points = [
        (0.0655, "20260713_p5_s-wave-d-control", "0.06M"),
        (0.262, "20260713_p5_s-wave-d-batch_025m", "0.25M"),
        (1.049, "20260713_p5_s-wave-d-batch_1m", "1M"),
    ]
    xs = [b[0] for b in batch_points]
    ys = [final_val(b[1]) for b in batch_points]
    labels = [b[2] for b in batch_points]
    ax_batch.plot(xs, ys, "o-", color="#4C6EF5", ms=9)
    for x, y, lab in zip(xs, ys, labels):
        ax_batch.annotate(f"{lab}\n({y:.2f})", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax_batch.set_xscale("log")
    ax_batch.set_title("Batch-size study (fixed ~98.3M token budget, lr NOT rescaled)")
    ax_batch.set_xlabel("effective batch (M tokens/step, log scale)"); ax_batch.set_ylabel("final val loss")

    # -- (d) grad-clip on vs off (train loss, early training) -------------------
    for label, run_id, color in [
        ("grad_clip=1.0 (control)", "20260713_p5_s-wave-d-control", "#4C6EF5"),
        ("grad_clip=1e6 (off)", "20260713_p5_s-wave-d-gradclip_off", "#E03131"),
    ]:
        with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
            recs = [json.loads(line) for line in f]
        steps = [r["step"] for r in recs]
        gn = [r["grad_norm"] for r in recs]
        ax_clip.plot(steps, gn, color=color, lw=1.4, label=f"{label} grad_norm")
    ax_clip.axhline(1.0, color="grey", lw=1, ls=":", label="clip threshold (control only)")
    ax_clip.set_title("Grad-clip on/off: raw grad_norm\n(clip_grad_norm_ always returns the PRE-clip norm)")
    ax_clip.set_xlabel("step"); ax_clip.set_ylabel("grad_norm"); ax_clip.legend(fontsize=8)

    fig.tight_layout()
    out = ROOT / "docs" / "results" / "wave_d_optimizers_schedules.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
