#!/usr/bin/env python
"""Wave F figure: DeepSeekMoE (aux-loss vs aux-loss-free balancing) + MTP.

Four panels from metrics.jsonl: (a) val_loss vs the dense control, (b)/(c) expert-load
heatmaps over training for each balancing method (the classic collapse-vs-balance picture),
(d) MTP's own auxiliary loss vs the main cross-entropy loss. Saved to
docs/results/wave_f_deepseek_specials.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = 0.015  # D-035 seed spread

CONTROL = "20260713_p5_s-wave-d-control"
MOE_AUXLOSS = "20260716_p5_s-wave-f-moe-auxloss"
MOE_BIASFREE = "20260716_p5_s-wave-f-moe-biasfree"
MTP = "20260716_p5_s-wave-f-mtp"


def records(run_id: str) -> list[dict]:
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def curve(run_id: str, key: str = "val_loss") -> tuple[list[float], list[float]]:
    toks, vals = [], []
    for d in records(run_id):
        if d.get(key) is not None:
            toks.append(d["tokens_seen"] / 1e6)
            vals.append(d[key])
    return toks, vals


def final_val(run_id: str) -> float:
    _, v = curve(run_id)
    return v[-1]


def expert_load_matrix(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, load) where load is (n_logged_steps, n_experts) -- averaged across layers."""
    steps, loads = [], []
    for d in records(run_id):
        if d.get("expert_load") is not None:
            steps.append(d["step"])
            per_layer = np.array(d["expert_load"])  # (n_layers, n_experts)
            loads.append(per_layer.mean(axis=0))
    return np.array(steps), np.array(loads)  # loads: (n_logged_steps, n_experts)


def main() -> None:
    fig = plt.figure(figsize=(13, 9))
    ax_loss = fig.add_subplot(2, 2, 1)
    ax_aux = fig.add_subplot(2, 2, 2)
    ax_hist_aux = fig.add_subplot(2, 2, 3)
    ax_hist_bias = fig.add_subplot(2, 2, 4)

    # -- (a) val_loss vs dense control -----------------------------------------
    control_final = final_val(CONTROL)
    for label, run_id, color in [
        ("dense control", CONTROL, "#4C6EF5"),
        ("MoE, aux_loss", MOE_AUXLOSS, "#F76707"),
        ("MoE, bias_free (V3)", MOE_BIASFREE, "#94D82D"),
        ("+MTP head", MTP, "#AE3EC9"),
    ]:
        t, v = curve(run_id)
        ax_loss.plot(t, v, label=f"{label} (final {v[-1]:.3f})", color=color, lw=1.8)
    ax_loss.axhspan(control_final - NOISE_FLOOR, control_final + NOISE_FLOOR, color="grey",
                     alpha=0.15, label=f"control ± noise floor ({NOISE_FLOOR})")
    ax_loss.set_title("val loss vs dense control (~98.3M token budget)")
    ax_loss.set_xlabel("tokens seen (M)"); ax_loss.set_ylabel("val loss"); ax_loss.legend(fontsize=8)

    # -- (b) moe_aux_loss trend (aux_loss run) vs main train_loss, + MTP's own loss --
    for label, run_id, key, color in [
        ("MoE aux_loss (balancing=aux_loss)", MOE_AUXLOSS, "moe_aux_loss", "#F76707"),
        ("MoE aux_loss (balancing=bias_free, should be ~0)", MOE_BIASFREE, "moe_aux_loss", "#94D82D"),
    ]:
        recs = [d for d in records(run_id) if d.get(key) is not None]
        steps = [d["step"] for d in recs]
        vals = [d[key] for d in recs]
        ax_aux.plot(steps, vals, color=color, lw=1.4, label=label)
    ax_aux_twin = ax_aux.twinx()
    mtp_recs = [d for d in records(MTP) if d.get("mtp_loss") is not None]
    ax_aux_twin.plot([d["step"] for d in mtp_recs], [d["mtp_loss"] for d in mtp_recs],
                      color="#AE3EC9", lw=1.4, ls="--", label="MTP head loss (right axis)")
    ax_aux.set_title("Balancing-loss trend + MTP head's own loss")
    ax_aux.set_xlabel("step"); ax_aux.set_ylabel("moe_aux_loss")
    ax_aux_twin.set_ylabel("mtp_loss")
    lines1, labels1 = ax_aux.get_legend_handles_labels()
    lines2, labels2 = ax_aux_twin.get_legend_handles_labels()
    ax_aux.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # -- (c)/(d) expert load heatmaps: the collapse-vs-balance picture ----------
    for ax, run_id, title in [
        (ax_hist_aux, MOE_AUXLOSS, "Expert load over training -- aux_loss"),
        (ax_hist_bias, MOE_BIASFREE, "Expert load over training -- bias_free (V3)"),
    ]:
        steps, load = expert_load_matrix(run_id)  # load: (n_steps, n_experts)
        im = ax.imshow(
            load.T, aspect="auto", origin="lower", cmap="viridis", vmin=0,
            extent=[steps.min(), steps.max(), -0.5, load.shape[1] - 0.5],
        )
        ax.set_title(title)
        ax.set_xlabel("step"); ax.set_ylabel("expert index")
        ax.set_yticks(range(load.shape[1]))
        fig.colorbar(im, ax=ax, label="fraction of tokens routed here")

    fig.tight_layout()
    out = ROOT / "docs" / "results" / "wave_f_deepseek_specials.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
