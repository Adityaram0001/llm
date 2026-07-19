"""Load a fine-tuned model from a phase-8 run folder, handling both full-FT and LoRA checkpoints.

A full-FT checkpoint carries `model_state_dict` (load directly). A LoRA checkpoint carries only
`lora_state_dict` + the `base_checkpoint` path + the `lora_config` it was trained with, so it is
reconstructed as: base weights → `apply_lora` (same targets/rank) → load the adapter tensors. Both
`scripts/eval_sft.py` and `scripts/chat.py` go through this so neither has to know which it got.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from llmlab.model import GPT, ModelConfig
from llmlab.utils import get_device

from .trainer import ROOT


def load_finetuned(
    run_dir: Path, ckpt_name: str = "best.pt", device: torch.device | None = None
) -> tuple[GPT, Tokenizer, dict]:
    """Returns `(model.eval(), tokenizer, run_config_dict)` for a phase-8 run folder."""
    device = device or get_device()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tokenizer = Tokenizer.from_file(str(ROOT / cfg["tokenizer_dir"] / "tokenizer.json"))
    model = GPT(ModelConfig.from_yaml(str(ROOT / cfg["model_config"]))).to(device)

    ckpt = torch.load(run_dir / "ckpt" / ckpt_name, map_location=device)
    if "lora_state_dict" in ckpt:
        from .lora import apply_lora, load_lora_state

        base = torch.load(ROOT / ckpt["base_checkpoint"], map_location=device)
        model.load_state_dict(base["model_state_dict"])
        lc = ckpt["lora_config"]
        apply_lora(model, lc["targets"], r=lc["r"], alpha=lc["alpha"], dropout=lc["dropout"])
        model.to(device)  # adapter params are created on CPU -> move to device before loading
        load_lora_state(model, ckpt["lora_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model, tokenizer, cfg
