"""Trainer: config-driven training loop shared by every phase-4/5 run.

Design note (D-021): there is no `torch.utils.data.DataLoader` here — `MixedSourceLoader`
(`llmlab.data.loader`) does direct memmap random-access sampling on CPU and hands back tensors
already moved to `device`, so there is nothing to parallelize with worker processes. This is the
standard nanoGPT-style pattern for a memmap corpus and is simpler than a DataLoader + Dataset
pair; `num_workers`/`pin_memory` config keys (CLOUD.md's general portability rule) are therefore
not applicable to this loader and are intentionally absent from `TrainConfig`.

**Why optimizer state roughly doubles checkpoint size:** AdamW keeps two running moments
(`exp_avg`, `exp_avg_sq`) per parameter, each the same shape/dtype as the parameter itself — so
the optimizer state alone is ~2x the model's own parameter count.

**Why resume needs so little state:** the loader is stateless given `(seed, step)` (see
loader.py) — replaying step N+1 onward after a resume reproduces the exact batches an
uninterrupted run would have produced, so the checkpoint only needs the model, optimizer, and
the integer step counter; there is no sampler/iterator state to save.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import signal
import time
from pathlib import Path

import torch
import wandb
from tokenizers import Tokenizer

from llmlab.data.loader import MixedSourceLoader, Source
from llmlab.model import GPT, ModelConfig
from llmlab.utils import autocast_ctx, get_device, mem_stats, set_seed

from .config import DataSourceConfig, TrainConfig

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "experiments" / "registry.csv"


def _sources(cfgs: list[DataSourceConfig]) -> list[Source]:
    return [
        Source(
            name=c.name,
            bin_path=ROOT / c.path,
            weight=c.weight,
            respect_doc_boundaries=c.respect_doc_boundaries,
            docstarts_path=(ROOT / c.docstarts_path) if c.docstarts_path else None,
        )
        for c in cfgs
    ]


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Two AdamW param groups: matrix weights get weight decay, everything else doesn't.

    Norm gains and embeddings are excluded because weight decay's "shrink toward zero"
    regularization doesn't make sense for them: a norm gain is a single learned scale/shift, not
    a projection whose magnitude trades off against overfitting; embeddings are a lookup table
    where decaying a rarely-seen token's row toward zero actively destroys its (already
    data-starved) representation rather than regularizing it. This model has no biases
    (`bias=False` throughout), so in practice the no-decay group is exactly {tok_emb, norms}.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name or "pos_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to `lr * lr_min_ratio`."""
    o = cfg.optim
    lr_min = o.lr * o.lr_min_ratio
    if step < o.warmup_steps:
        return o.lr * (step + 1) / o.warmup_steps
    if step >= cfg.max_steps:
        return lr_min
    progress = (step - o.warmup_steps) / max(1, cfg.max_steps - o.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + coeff * (o.lr - lr_min)


class Trainer:
    def __init__(self, cfg: TrainConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.device = torch.device(cfg.device) if cfg.device else get_device()
        set_seed(cfg.seed)

        model_cfg = ModelConfig.from_yaml(str(ROOT / cfg.model_config))
        self.model = GPT(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.optim.weight_decay),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
        )

        self.train_loader = MixedSourceLoader(_sources(cfg.sources), cfg.seq_len, cfg.seed)
        self.val_loader = MixedSourceLoader(_sources(cfg.val_sources), cfg.seq_len, cfg.seed + 1)
        self._eval_batches = self.val_loader.fixed_eval_batches(
            cfg.eval.eval_batches, cfg.eval.eval_batch_size, self.device
        )

        self.tokenizer = Tokenizer.from_file(str(ROOT / cfg.tokenizer_dir / "tokenizer.json"))

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.tokens_per_step = cfg.batch.micro_batch * cfg.batch.grad_accum * cfg.seq_len
        self._start_time = time.time()

        self._wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            name=run_dir.name,
            config=cfg.to_dict(),
            mode=cfg.logging.wandb_mode,
            dir=str(run_dir),
        )
        # wandb.init() installs its own SIGINT handler (to flush its writer on Ctrl-C) that
        # swallows the interrupt instead of letting it raise KeyboardInterrupt in the main
        # thread -- silently breaking fit()'s graceful-Ctrl-C handling below. Restore the
        # default handler (kill -INT / Ctrl-C -> KeyboardInterrupt) so `except KeyboardInterrupt`
        # actually fires. Found by testing the real resume flow (D-021 session): a plain
        # `kill -INT <pid>` was ignored entirely until this line was added.
        signal.signal(signal.SIGINT, signal.default_int_handler)

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val_loss = ckpt["best_val_loss"]

    # -- core loop --------------------------------------------------------

    def train_step(self) -> tuple[float, float, float]:
        self.model.train()
        lr = lr_at_step(self.step, self.cfg)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for micro in range(self.cfg.batch.grad_accum):
            data_step = self.step * self.cfg.batch.grad_accum + micro
            x, y = self.train_loader.get_batch(data_step, self.cfg.batch.micro_batch, self.device)
            with autocast_ctx(self.device):
                _, loss = self.model(x, y)
            loss = loss / self.cfg.batch.grad_accum
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optim.grad_clip
        )
        self.optimizer.step()
        self.tokens_seen += self.tokens_per_step
        return total_loss, float(grad_norm), lr

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = [self.model(x, y)[1].item() for x, y in self._eval_batches]
        self.model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def generate_samples(self) -> None:
        lines = []
        for prompt in self.cfg.sampling.prompts:
            ids = self.tokenizer.encode(prompt).ids
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            out = self.model.generate(
                idx, max_new_tokens=self.cfg.sampling.max_new_tokens, temperature=0.8, top_k=40
            )
            lines.append(f"--- prompt: {prompt!r} ---\n{self.tokenizer.decode(out[0].tolist())}\n")
        out_path = self.run_dir / "samples" / f"step_{self.step:06d}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def _log(self, step: int, train_loss: float, grad_norm: float, lr: float, tokens_per_sec: float, val_loss: float | None) -> None:
        record = {
            "step": step,
            "tokens_seen": self.tokens_seen,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "mem_gb": mem_stats()["rss_mb"] / 1024,
            "elapsed_s": time.time() - self._start_time,
        }
        if val_loss is not None:
            record["val_loss"] = val_loss
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        wandb.log(record, step=step)

    def fit(self) -> str:
        """Run until `max_steps`. Returns a status string ("completed" or "interrupted").

        `self.step` means "how many steps have fully completed" (== the next step to run) at
        every point where a checkpoint could be written -- it is only bumped from `step` to
        `step + 1` *after* a step's eval/log/sample work is done. This matters for resume: an
        earlier version bumped `self.step` via the for-loop's own iteration variable, so a
        checkpoint taken between the loop's last statement and its next iteration (exactly
        where a Ctrl-C lands) saved the *just-completed* step instead of the next one, silently
        re-running (and re-applying the gradient update for) that step on resume. Caught by
        actually killing and resuming a real run rather than trusting the unit test alone: the
        replayed step's logged loss differed from its first run, which shouldn't be possible
        with a stateless, (seed, step)-keyed loader unless the model weights underneath it had
        already moved (see docs/DECISIONS.md).
        """
        status = "completed"
        try:
            from tqdm import tqdm

            start_step = self.step
            pbar = tqdm(range(start_step, self.cfg.max_steps), initial=start_step, total=self.cfg.max_steps)
            for step in pbar:
                self.step = step  # train_step()/lr_at_step read this for the step about to run
                t0 = time.time()
                train_loss, grad_norm, lr = self.train_step()
                step_time = time.time() - t0
                tokens_per_sec = self.tokens_per_step / step_time

                val_loss = None
                if step % self.cfg.eval.eval_every == 0:
                    val_loss = self.evaluate()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(self.run_dir / "ckpt" / "best.pt")

                if step % self.cfg.logging.log_every == 0 or val_loss is not None:
                    self._log(step, train_loss, grad_norm, lr, tokens_per_sec, val_loss)
                    pbar.set_postfix(loss=f"{train_loss:.3f}", val=f"{val_loss:.3f}" if val_loss else "-", lr=f"{lr:.2e}")

                if step % self.cfg.sampling.sample_every == 0:
                    self.generate_samples()

                self.step = step + 1  # `step` is now fully done -- safe resume point
                if self.step % self.cfg.checkpoint_every == 0:
                    self.save_checkpoint(self.run_dir / "ckpt" / "latest.pt")
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            self.save_checkpoint(self.run_dir / "ckpt" / "latest.pt")
            self._append_registry_row(status)
            wandb.finish()
        return status

    # -- registry --------------------------------------------------------

    def _append_registry_row(self, status: str) -> None:
        row = [
            self.run_dir.name,
            datetime.date.today().isoformat(),
            self.cfg.phase,
            self.cfg.tier,
            round(self.model.num_params() / 1e6, 2),
            self.cfg.baseline_run,
            self.cfg.variable_changed,
            round(self.tokens_seen / 1e6, 2),
            round(self.best_val_loss, 4) if self.best_val_loss != float("inf") else "-",
            round(math.exp(self.best_val_loss), 2) if self.best_val_loss != float("inf") else "-",
            round((time.time() - self._start_time) / 3600, 3),
            self._wandb_run.url if self._wandb_run and self.cfg.logging.wandb_mode == "online" else "-",
            f"{status} at step {self.step}/{self.cfg.max_steps} -- review and fill in notes.md",
        ]
        with REGISTRY_PATH.open("a", newline="") as f:
            csv.writer(f).writerow(row)
