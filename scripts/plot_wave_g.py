#!/usr/bin/env python
"""Wave G figure: domain-mix ablation (RW-4), multi-epoch overfitting lab, and the mini
scaling law -- three panels, saved to docs/results/wave_g_data_scaling.png. Also writes the
scaling-law power-law fit (L(N) = a*N^-alpha + c) to stdout for docs/results/recipe.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def curve(run_id: str, key: str = "val_loss") -> tuple[list[float], list[float]]:
    toks, vals = [], []
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get(key) is not None:
                toks.append(d["tokens_seen"] / 1e6)
                vals.append(d[key])
    return toks, vals


def train_curve(run_id: str) -> tuple[list[float], list[float]]:
    toks, vals = [], []
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("train_loss") is not None:
                toks.append(d["tokens_seen"] / 1e6)
                vals.append(d["train_loss"])
    return toks, vals


def final_val(run_id: str) -> float:
    return curve(run_id)[1][-1]


def best_val(run_id: str) -> tuple[float, float]:
    """(best val_loss, tokens_seen_M at that point) -- the early-stopped point, not the last
    logged step. At this wave's fixed 200M-token budget (~11.3 epochs over the 17.66M-token
    pool), the two larger scaling-law models overfit the repeated corpus well before the budget
    ends (val loss bottoms out then rises even as train loss keeps falling) -- "final" and "best"
    diverge sharply for them, so the scaling-law fit uses "best" as the fairer per-N comparison."""
    toks, vals = curve(run_id)
    i = min(range(len(vals)), key=lambda i: vals[i])
    return vals[i], toks[i]


def power_law(n, a, alpha, c):
    return a * np.power(n, -alpha) + c


def fit_power_law(ns: np.ndarray, ls: np.ndarray) -> tuple[float, float, float, float]:
    """Fit L(N) = a*N^-alpha + c without a scipy dependency.

    Grid-search the irreducible-loss constant `c` (must be < min(ls)); for each candidate,
    log(L - c) = log(a) - alpha*log(N) is LINEAR, solved by `np.polyfit` in log-log space.
    Picks the `c` minimizing total squared residual in loss-space (not log-space) so the fit
    is judged on the scale we actually plot/report.
    """
    log_n = np.log(ns)
    best = None
    for c in np.linspace(0.0, ls.min() * 0.98, 400):
        residual = ls - c
        if (residual <= 0).any():
            continue
        slope, intercept = np.polyfit(log_n, np.log(residual), 1)
        alpha, a = -slope, np.exp(intercept)
        pred = power_law(ns, a, alpha, c)
        sse = float(np.sum((pred - ls) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, alpha, c)
    _, a, alpha, c = best
    return a, alpha, c, best[0]


def main() -> None:
    fig, (ax_mix, ax_epoch, ax_scale) = plt.subplots(1, 3, figsize=(17, 5))

    # -- (a) domain-mix ablation: general val loss vs domain share -----------------
    mix_runs = [
        (0, "20260716_p5_s-wave-g-domainmix-00"),
        (10, "20260716_p5_s-wave-g-domainmix-10"),
        (25, "20260716_p5_s-wave-g-domainmix-25"),
        (50, "20260716_p5_s-wave-g-domainmix-50"),
    ]
    shares = [m[0] for m in mix_runs]
    losses = [final_val(m[1]) for m in mix_runs]
    ax_mix.plot(shares, losses, "o-", color="#4C6EF5", lw=1.8)
    for s, l in zip(shares, losses):
        ax_mix.annotate(f"{l:.3f}", (s, l), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax_mix.set_title("Domain-mix ablation (RW-4)\nfinance/self-help/wisdom share -> general val loss")
    ax_mix.set_xlabel("domain share of training stream (%)")
    ax_mix.set_ylabel("val loss (general books+dictionary val)")

    # -- (b) multi-epoch overfitting lab: train/val gap vs epochs -------------------
    epoch_runs = [
        (1, "20260716_p5_s-wave-g-epochs-01"),
        (4, "20260716_p5_s-wave-g-epochs-04"),
        (16, "20260716_p5_s-wave-g-epochs-16"),
    ]
    colors = ["#4C6EF5", "#F76707", "#E03131"]
    for (n_epochs, run_id), color in zip(epoch_runs, colors):
        t_tr, v_tr = train_curve(run_id)
        t_va, v_va = curve(run_id)
        # x-axis in EPOCHS (tokens seen / pool size), not raw tokens, so all 3 lines share a scale
        pool_m = 14.141233
        ax_epoch.plot([t / pool_m for t in t_tr], v_tr, color=color, lw=1.2, ls="--", alpha=0.6)
        ax_epoch.plot([t / pool_m for t in t_va], v_va, color=color, lw=2.0, label=f"{n_epochs} epochs (val, final gap {v_va[-1]-v_tr[-1]:+.3f})")
    ax_epoch.set_title("Multi-epoch overfitting lab (books-only pool)\ndashed=train, solid=val")
    ax_epoch.set_xlabel("epochs over the 14.14M-token books-only pool")
    ax_epoch.set_ylabel("loss")
    ax_epoch.legend(fontsize=8)

    # -- (c) mini scaling law: val loss vs param count (log-log) --------------------
    scale_runs = [
        (4_999_168, "20260716_p5_scaling-5m"),
        (9_713_472, "20260716_p5_scaling-10m"),
        (24_786_240, "20260716_p5_scaling-25m"),
        (50_400_384, "20260716_p5_scaling-50m"),
    ]
    ns = np.array([r[0] for r in scale_runs], dtype=float)
    best = [best_val(r[1]) for r in scale_runs]
    ls = np.array([b[0] for b in best])          # early-stopped -- the fit target
    ls_final = np.array([final_val(r[1]) for r in scale_runs])  # raw end-of-budget number

    # final (end-of-budget) numbers as faded x's, to visualize how far overfitting drags the
    # bigger models away from their own best-so-far point
    ax_scale.plot(ns, ls_final, "x", color="#E03131", ms=8, zorder=2, label="final (step 3050, overfit for N>=25M)")
    ax_scale.plot(ns, ls, "o", color="#4C6EF5", ms=8, zorder=3, label="best (early-stopped)")
    for n, l in zip(ns, ls):
        ax_scale.annotate(f"{l:.3f}", (n, l), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")

    a, alpha, c, sse = fit_power_law(ns, ls)
    n_grid = np.logspace(np.log10(ns.min()), np.log10(ns.max()), 100)
    ax_scale.plot(n_grid, power_law(n_grid, a, alpha, c), "-", color="#868E96", lw=1.5, zorder=1,
                  label=f"fit (on best): L={a:.2f}*N^-{alpha:.3f}+{c:.3f}")
    print(f"Scaling law fit (on BEST/early-stopped val_loss): L(N) = {a:.4f} * N^-{alpha:.4f} + {c:.4f}  (sse={sse:.5f}; Chinchilla alpha ~= 0.34)")

    ax_scale.set_xscale("log")
    ax_scale.set_title("Mini scaling law (fixed 200M tokens ~= 11.3 epochs)\nbigger models overfit the repeated pool before the budget ends")
    ax_scale.set_xlabel("non-embedding+embedding params N (log)")
    ax_scale.set_ylabel("val loss")
    ax_scale.legend(fontsize=7)

    fig.tight_layout()
    out = ROOT / "docs" / "results" / "wave_g_data_scaling.png"
    fig.savefig(out, dpi=140)
    print("saved", out)

    print("\n=== domain-mix ===")
    for s, l in zip(shares, losses):
        print(f"  {s:3d}% -> val_loss={l:.4f}")
    print("\n=== multi-epoch ===")
    for n_epochs, run_id in epoch_runs:
        t_tr, v_tr = train_curve(run_id)
        t_va, v_va = curve(run_id)
        print(f"  {n_epochs:2d} epochs -> train={v_tr[-1]:.4f} val={v_va[-1]:.4f} gap={v_va[-1]-v_tr[-1]:+.4f}")
    print("\n=== scaling law ===")
    for (n, run_id), (b, tok_at_best), lf in zip(scale_runs, best, ls_final):
        overfit_gap = lf - b
        print(f"  N={n/1e6:6.2f}M -> best_val={b:.4f} @ {tok_at_best:.1f}M tok   final_val={lf:.4f}   overfit_gap={overfit_gap:+.4f}")


if __name__ == "__main__":
    main()
