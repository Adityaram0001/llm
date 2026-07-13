"""TrainConfig: the config-driven surface for `src/llmlab/train/trainer.py`.

One YAML file (`configs/train_*.yaml`) fully determines a run — no hyperparameters hard-coded
in scripts, per CLAUDE.md's experiment-discipline rule. Registry metadata (`phase`, `tier`,
`baseline_run`, `variable_changed`) lives here too so `scripts/train.py` can append a
`experiments/registry.csv` row without extra bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataSourceConfig:
    name: str
    path: str
    weight: float = 1.0
    respect_doc_boundaries: bool = False
    docstarts_path: str | None = None


@dataclass
class OptimConfig:
    lr: float
    lr_min_ratio: float = 0.1  # floor of the schedule = lr * lr_min_ratio
    warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Wave D (phase 5): optimizer choice + hybrid-optimizer (Muon) knobs
    optimizer: str = "adamw"  # adamw | lion | muon
    muon_lr: float = 0.02  # Muon's own peak lr (only used when optimizer="muon")
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # Wave D: lr schedule shape + z-loss
    schedule: str = "cosine"  # cosine | wsd | constant
    wsd_decay_ratio: float = 0.2  # last X fraction of steps decay to lr_min (schedule="wsd")
    z_loss_weight: float = 0.0  # PaLM '22 z-loss coefficient; 0 disables


@dataclass
class BatchConfig:
    micro_batch: int
    grad_accum: int


@dataclass
class EvalConfig:
    eval_every: int
    eval_batches: int
    eval_batch_size: int


@dataclass
class SamplingConfig:
    sample_every: int
    max_new_tokens: int = 80
    prompts: list[str] = field(
        default_factory=lambda: ["Once upon a time", "ephemeral (adjective):"]
    )


@dataclass
class LoggingConfig:
    log_every: int = 10
    wandb_project: str = "llm-lab"
    wandb_mode: str = "offline"  # D-009: offline by default, `wandb sync` later if wanted


@dataclass
class TrainConfig:
    seed: int
    model_config: str
    tokenizer_dir: str
    seq_len: int
    sources: list[DataSourceConfig]
    val_sources: list[DataSourceConfig]
    optim: OptimConfig
    batch: BatchConfig
    max_steps: int
    eval: EvalConfig
    sampling: SamplingConfig
    logging: LoggingConfig
    device: str | None = None  # None -> get_device(); "cpu" for the portability smoke test
    checkpoint_every: int = 500

    # experiment-registry metadata (docs/EXPERIMENTS.md schema)
    phase: int = 4
    tier: str = "S"
    baseline_run: str = "-"
    variable_changed: str = "-"

    def __post_init__(self) -> None:
        self.sources = [
            s if isinstance(s, DataSourceConfig) else DataSourceConfig(**s) for s in self.sources
        ]
        self.val_sources = [
            s if isinstance(s, DataSourceConfig) else DataSourceConfig(**s)
            for s in self.val_sources
        ]
        if not isinstance(self.optim, OptimConfig):
            self.optim = OptimConfig(**self.optim)
        self.optim.betas = tuple(self.optim.betas)
        if not isinstance(self.batch, BatchConfig):
            self.batch = BatchConfig(**self.batch)
        if not isinstance(self.eval, EvalConfig):
            self.eval = EvalConfig(**self.eval)
        if not isinstance(self.sampling, SamplingConfig):
            self.sampling = SamplingConfig(**self.sampling)
        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig(**self.logging)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        """Resolved config as plain JSON-safe nested dicts (tuples -> lists), for dumping to
        the run folder as `config.yaml` -- the exact frozen record of what a run used."""
        import json
        from dataclasses import asdict

        return json.loads(json.dumps(asdict(self)))
