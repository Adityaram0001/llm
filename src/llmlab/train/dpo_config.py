"""DPOConfig: the config-driven surface for `dpo_trainer.py` (phase 8, Part C).

Separate from `SFTConfig` because DPO's data shape (paired chosen/rejected, not single
instruction/response) and its loss (needs a frozen reference model alongside the trainable
policy) are genuinely different — but the surrounding machinery (warm start, epoch loop over a
finite set, catastrophic-forgetting probe, registry row) is deliberately the same shape as
`SFTConfig` so the three phase-8 configs read as one family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import LoggingConfig


@dataclass
class DPOConfig:
    seed: int
    model_config: str  # must match the SFT run's architecture
    tokenizer_dir: str
    sft_run: str  # experiments/<sft run folder> -- both the policy's init AND the frozen reference
    sft_ckpt_name: str = "best.pt"
    train_file: str = "data/dpo/dictionary_pairs/train.jsonl"
    val_file: str = "data/dpo/dictionary_pairs/val.jsonl"

    # pretrain val set, for the catastrophic-forgetting probe (same mechanic as SFTConfig)
    pretrain_val_bin: str = "data/tokenized/hf_bpe_16k/val.bin"
    pretrain_val_seq_len: int = 512
    pretrain_val_batches: int = 16
    pretrain_val_batch_size: int = 16

    max_len: int = 640  # bigger than SFT's 128: the deliberately-verbose rejected side runs long
                         # (measured p99=524/max=607 tokens on the real dictionary_pairs set)
    epochs: int = 1  # DPO overfits/over-drifts fast on a finite, un-augmented pair set (RW-6-style caution)
    batch_size: int = 16  # two forward passes (policy+ref) x two sides (chosen+rejected) per step
    supervise_eot: bool = True

    beta: float = 0.1  # DPO temperature -- see dpo.py's docstring for what it trades off

    lr: float = 5.0e-6  # DPO papers run 10-100x lower than SFT lr; drift is the risk, not underfitting
    lr_min_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    eval_every: int = 25
    sample_every: int = 25
    sample_prompts: list[str] = field(
        default_factory=lambda: ["What does ephemeral mean?", "Define 'philosophy'."]
    )

    device: str | None = None
    precision: str = "bf16"
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # experiment-registry metadata (docs/EXPERIMENTS.md schema)
    phase: int = 8
    tier: str = "S"
    baseline_run: str = "-"
    variable_changed: str = "-"

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)
        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig(**self.logging)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        import json
        from dataclasses import asdict

        return json.loads(json.dumps(asdict(self)))
