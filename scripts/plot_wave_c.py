#!/usr/bin/env python
"""Wave C figure: val-loss curves + the cache-size-vs-quality tradeoff scatter.

Reads the four attention-variant runs' metrics.jsonl and pairs each final val loss with its
analytical KV-cache bytes/token/layer, producing docs/results/wave_c_attention_variants.png.
The right panel is the point of the whole wave: does the cheaper cache cost you quality?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from llmlab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]

# (label, run_id, model_config, color)
RUNS = [
    ("MHA (control)", "20260713_p5_s-wave-c-mha", "configs/model_s_attn_mha.yaml", "#4C6EF5"),
    ("GQA-2", "20260713_p5_s-wave-c-gqa2", "configs/model_s_attn_gqa2.yaml", "#22B8CF"),
    ("MQA", "20260713_p5_s-wave-c-mqa", "configs/model_s_attn_mqa.yaml", "#94D82D"),
    ("MLA", "20260713_p5_s-wave-c-mla", "configs/model_s_attn_mla.yaml", "#F76707"),
]
NOISE_FLOOR = 0.015  # D-035 seed spread


def cache_bytes_per_tok_layer(cfg: ModelConfig, dtype_size: int = 2) -> int:
    if cfg.attention == "mla":
        return (cfg.mla.kv_lora_rank + cfg.mla.rope_head_dim) * dtype_size
    return 2 * cfg.n_kv_heads * cfg.head_dim * dtype_size


def load_curve(run_id: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    with open(ROOT / "experiments" / run_id / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("val_loss") is not None:
                steps.append(d["step"])
                vals.append(d["val_loss"])
    return steps, vals


def main() -> None:
    fig, (axc, axt) = plt.subplots(1, 2, figsize=(12, 4.6))

    finals = {}
    for label, run_id, cfg_path, color in RUNS:
        cfg = ModelConfig.from_yaml(str(ROOT / cfg_path))
        steps, vals = load_curve(run_id)
        axc.plot(steps, vals, label=label, color=color, lw=1.8)
        finals[label] = (cache_bytes_per_tok_layer(cfg), vals[-1], color)

    # left: val-loss curves
    mha_final = finals["MHA (control)"][1]
    axc.axhspan(mha_final - NOISE_FLOOR, mha_final + NOISE_FLOOR, color="grey", alpha=0.15,
                label=f"MHA ± noise floor ({NOISE_FLOOR})")
    axc.set_xlabel("step"); axc.set_ylabel("val loss")
    axc.set_title("Wave C — val loss over training")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)

    # right: cache-bytes vs final val loss (the tradeoff)
    for label, (cb, fv, color) in finals.items():
        axt.scatter(cb, fv, s=120, color=color, zorder=3)
        axt.annotate(f"{label}\n{cb} B/tok/layer\nval {fv:.4f}", (cb, fv),
                     textcoords="offset points", xytext=(8, 6), fontsize=8)
    axt.axhspan(mha_final - NOISE_FLOOR, mha_final + NOISE_FLOOR, color="grey", alpha=0.15)
    axt.set_xlabel("KV cache bytes / token / layer (bf16)  →  cheaper is left")
    axt.set_ylabel("final val loss  →  better is down")
    axt.set_title("Wave C — cache size vs quality tradeoff", pad=18)
    axt.grid(alpha=0.3)
    _lo = min(fv for _, fv, _ in finals.values())
    _hi = max(fv for _, fv, _ in finals.values())
    axt.set_ylim(_lo - 0.006, _hi + 0.010)  # headroom so top annotation clears the title

    plt.tight_layout()
    out = ROOT / "docs/results/wave_c_attention_variants.png"
    plt.savefig(out, dpi=130)
    print("wrote", out.relative_to(ROOT))
    print("\nfinal val losses:")
    for label, (cb, fv, _) in finals.items():
        print(f"  {label:15s} cache={cb:>4d} B  val={fv:.4f}  delta_vs_MHA={fv - mha_final:+.4f}")


if __name__ == "__main__":
    main()
