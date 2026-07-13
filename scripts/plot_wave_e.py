#!/usr/bin/env python
"""Wave E figure: bf16 vs fp32, gradient checkpointing (loss parity + the real memory-vs-seq_len
curve from bench_activation_memory.py), micro-batch/accum equivalence, weight tying, and
tokens/sec across all of Wave E's runs (torch.compile included) -- five panels, saved to
docs/results/wave_e_efficiency_memory.png."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = 0.015  # D-035 seed spread

CONTROL = "20260713_p5_s-wave-d-control"


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
    return curve(run_id)[1][-1]


def mean_tokens_per_sec(run_id: str) -> float:
    vals = []
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("tokens_per_sec") is not None:
                vals.append(d["tokens_per_sec"])
    # drop the first few logged points -- they include one-time overhead (cudnn autotune,
    # torch.compile's graph capture) that isn't representative of steady-state throughput
    vals = vals[2:] if len(vals) > 4 else vals
    return sum(vals) / len(vals)


def read_actmem_csv(path: Path) -> tuple[list[int], list[float]]:
    seq_lens, peaks = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            seq_lens.append(int(row["seq_len"]))
            peaks.append(float(row["peak_mem_mb"]) / 1024)
    return seq_lens, peaks


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_prec, ax_ckpt_loss, ax_ckpt_mem, ax_batch, ax_tie, ax_speed = axes.flat

    # -- (a) bf16 vs fp32 loss curves --------------------------------------------
    for label, run_id, color in [
        ("bf16 (control)", CONTROL, "#4C6EF5"),
        ("fp32", "20260713_p5_s-wave-e-fp32", "#F76707"),
    ]:
        t, v = curve(run_id)
        ax_prec.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_prec.set_title("Precision: bf16 autocast vs fp32")
    ax_prec.set_xlabel("tokens seen (M)"); ax_prec.set_ylabel("val loss"); ax_prec.legend(fontsize=8)

    # -- (b) gradient checkpointing: loss parity ---------------------------------
    for label, run_id, color in [
        ("no checkpointing (control)", CONTROL, "#4C6EF5"),
        ("gradient checkpointing", "20260713_p5_s-wave-e-gradckpt", "#F76707"),
    ]:
        t, v = curve(run_id)
        ax_ckpt_loss.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_ckpt_loss.set_title("Gradient checkpointing: loss parity check\n(recompute, not an approximation -- should overlay)")
    ax_ckpt_loss.set_xlabel("tokens seen (M)"); ax_ckpt_loss.set_ylabel("val loss"); ax_ckpt_loss.legend(fontsize=8)

    # -- (c) gradient checkpointing: the real memory story -----------------------
    plain_csv = ROOT / "docs" / "results" / "wave_e_activation_memory.csv"
    ckpt_csv = ROOT / "docs" / "results" / "wave_e_activation_memory_gradckpt.csv"
    if plain_csv.exists() and ckpt_csv.exists():
        s1, m1 = read_actmem_csv(plain_csv)
        s2, m2 = read_actmem_csv(ckpt_csv)
        ax_ckpt_mem.plot(s1, m1, "o-", label="no checkpointing", color="#4C6EF5")
        ax_ckpt_mem.plot(s2, m2, "o-", label="gradient checkpointing", color="#F76707")
        ax_ckpt_mem.set_xscale("log", base=2)
        ax_ckpt_mem.set_title("Peak GPU memory vs seq_len (fwd+bwd, fixed micro_batch)")
        ax_ckpt_mem.set_xlabel("seq_len (log2)"); ax_ckpt_mem.set_ylabel("peak memory (GB)")
        ax_ckpt_mem.legend(fontsize=8)
    else:
        ax_ckpt_mem.text(0.5, 0.5, "run bench_activation_memory.py first", ha="center", va="center")

    # -- (d) micro-batch / grad-accum equivalence ---------------------------------
    for label, run_id, color in [
        ("mb=64,accum=2 (control)", CONTROL, "#4C6EF5"),
        ("mb=32,accum=4", "20260713_p5_s-wave-e-mb32_accum4", "#F76707"),
        ("mb=128,accum=1", "20260713_p5_s-wave-e-mb128_accum1", "#94D82D"),
    ]:
        t, v = curve(run_id)
        ax_batch.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_batch.set_title("Micro-batch / grad-accum equivalence\n(same effective batch=128 seqs/step, three factorizations)")
    ax_batch.set_xlabel("tokens seen (M)"); ax_batch.set_ylabel("val loss"); ax_batch.legend(fontsize=8)

    # -- (e) weight tying on vs off -------------------------------------------------
    for label, run_id, color in [
        ("tied (control, 9.71M)", CONTROL, "#4C6EF5"),
        ("untied (12.78M)", "20260713_p5_s-wave-e-notie", "#F76707"),
    ]:
        t, v = curve(run_id)
        ax_tie.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_tie.set_title("Weight tying: on vs off (+3.07M params untied)")
    ax_tie.set_xlabel("tokens seen (M)"); ax_tie.set_ylabel("val loss"); ax_tie.legend(fontsize=8)

    # -- (f) tokens/sec across the wave (incl. torch.compile) ------------------------
    speed_runs = [
        ("bf16\n(control)", CONTROL),
        ("fp32", "20260713_p5_s-wave-e-fp32"),
        ("grad\nckpt", "20260713_p5_s-wave-e-gradckpt"),
        ("mb=32\naccum=4", "20260713_p5_s-wave-e-mb32_accum4"),
        ("mb=128\naccum=1", "20260713_p5_s-wave-e-mb128_accum1"),
        ("untied", "20260713_p5_s-wave-e-notie"),
        ("torch\ncompile", "20260713_p5_s-wave-e-compile"),
    ]
    labels = [r[0] for r in speed_runs]
    speeds = [mean_tokens_per_sec(r[1]) for r in speed_runs]
    colors = ["#4C6EF5"] + ["#868E96"] * (len(speed_runs) - 2) + ["#F76707"]
    ax_speed.bar(labels, speeds, color=colors)
    ax_speed.set_title("Steady-state tokens/sec across Wave E")
    ax_speed.set_ylabel("tokens/sec"); ax_speed.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    out = ROOT / "docs" / "results" / "wave_e_efficiency_memory.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
