"""SFTConfig: the config-driven surface for `src/llmlab/train/sft_trainer.py` (phase 8, Part A).

Kept separate from `TrainConfig` (pretraining) because SFT's knobs are genuinely different — it
initializes from a pretrained checkpoint, iterates a finite example set in epochs (not an infinite
memmap stream), and its loss is assistant-token-masked. Shared registry metadata fields mirror
`TrainConfig` so the same `experiments/registry.csv` schema applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import LoggingConfig


@dataclass
class SFTConfig:
    seed: int
    model_config: str  # configs/model_s.yaml — must match the base checkpoint's architecture
    tokenizer_dir: str
    base_checkpoint: str  # experiments/<run>/ckpt/best.pt to initialize weights from
    train_file: str  # data/sft/<task>/train.jsonl
    val_file: str  # data/sft/<task>/val.jsonl

    # pretrain val set, for the catastrophic-forgetting probe (measured with plain CE, seq_len below)
    pretrain_val_bin: str
    pretrain_val_seq_len: int = 512
    pretrain_val_batches: int = 16
    pretrain_val_batch_size: int = 16

    max_len: int = 128
    epochs: int = 3
    batch_size: int = 32
    supervise_eot: bool = True

    lr: float = 2e-5
    lr_min_ratio: float = 0.1
    warmup_ratio: float = 0.03  # fraction of total optimizer steps spent warming up
    weight_decay: float = 0.0  # SFT is short; regularization usually off/low
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    eval_every: int = 25  # steps between val-loss + forgetting-probe measurements
    sample_every: int = 25
    sample_prompts: list[str] = field(
        default_factory=lambda: ["What does ephemeral mean?", "Define 'philosophy'."]
    )

    device: str | None = None
    precision: str = "bf16"  # bf16 (autocast) | fp32
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
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        import json
        from dataclasses import asdict

        return json.loads(json.dumps(asdict(self)))
