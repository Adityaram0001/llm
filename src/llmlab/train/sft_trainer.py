"""SFTTrainer: supervised fine-tuning loop (phase 8, Part A).

Distinct from the pretrain `Trainer` (`trainer.py`) in three ways, all of which are the point of
the phase:
  1. **Warm start** — weights come from a pretrained checkpoint (`SFTConfig.base_checkpoint`),
     not random init. A fresh AdamW is built (optimizer state from pretraining is irrelevant to a
     new objective at a new, much lower lr).
  2. **Assistant-masked loss** — batches come from `SFTDataset` with non-assistant targets set to
     the ignore index (`sft_loader.py`); the model's `cross_entropy(ignore_index=-1)` does the rest.
  3. **Finite epochs + a forgetting probe** — the loop runs `epochs` passes over a small example
     set, and at every eval it also measures the frozen pretrain-val perplexity. Watching SFT loss
     fall while pretrain ppl rises is catastrophic forgetting made visible — the core lesson.

Reuses `trainer.build_param_groups` (no weight decay on norms/embeddings), the checkpoint format,
and the registry-row convention so SFT runs land in `experiments/` like every other run.
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
from llmlab.data.sft_loader import SFTDataset
from llmlab.model import GPT, ModelConfig
from llmlab.utils import autocast_ctx, get_device, mem_stats, set_seed

from .sft_config import SFTConfig
from .trainer import ROOT, REGISTRY_PATH, build_param_groups


def sft_lr_at_step(step: int, total_steps: int, warmup_steps: int, cfg: SFTConfig) -> float:
    """Linear warmup → cosine decay to `lr * lr_min_ratio` over `total_steps` (the standard SFT
    schedule). Separate from the pretrain schedule because SFT is defined in epochs, so the total
    step count is known up front (epochs × batches/epoch) rather than a config field."""
    if warmup_steps > 0 and step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.lr_min_ratio + coeff * (1 - cfg.lr_min_ratio))


class SFTTrainer:
    def __init__(self, cfg: SFTConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.device = torch.device(cfg.device) if cfg.device else get_device()
        set_seed(cfg.seed)

        model_cfg = ModelConfig.from_yaml(str(ROOT / cfg.model_config))
        self.model = GPT(model_cfg).to(self.device)
        self._load_base_weights(ROOT / cfg.base_checkpoint)

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas
        )

        self.tokenizer = Tokenizer.from_file(str(ROOT / cfg.tokenizer_dir / "tokenizer.json"))
        self.train_ds = SFTDataset.from_jsonl(
            ROOT / cfg.train_file, self.tokenizer, max_len=cfg.max_len, supervise_eot=cfg.supervise_eot
        )
        self.val_ds = SFTDataset.from_jsonl(
            ROOT / cfg.val_file, self.tokenizer, max_len=cfg.max_len, supervise_eot=cfg.supervise_eot
        )
        self.val_batches = self.val_ds.eval_batches(cfg.batch_size, self.device)

        # Frozen pretrain-val batches for the catastrophic-forgetting probe (plain CE, no mask).
        pretrain_val = MixedSourceLoader(
            [Source(name="pretrain_val", bin_path=ROOT / cfg.pretrain_val_bin, weight=1.0)],
            cfg.pretrain_val_seq_len,
            cfg.seed + 1,
        )
        self.pretrain_val_batches = pretrain_val.fixed_eval_batches(
            cfg.pretrain_val_batches, cfg.pretrain_val_batch_size, self.device
        )

        self.steps_per_epoch = math.ceil(len(self.train_ds) / cfg.batch_size)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.warmup_steps = int(cfg.warmup_ratio * self.total_steps)

        self.step = 0
        self.best_val_loss = float("inf")
        self.pretrain_val_loss_0 = None  # baseline forgetting reference, filled on first eval
        self._start_time = time.time()

        self._wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            name=run_dir.name,
            config=cfg.to_dict(),
            mode=cfg.logging.wandb_mode,
            dir=str(run_dir),
        )
        signal.signal(signal.SIGINT, signal.default_int_handler)  # see Trainer (D-023)

    # -- setup helpers --------------------------------------------------------

    def _load_base_weights(self, path: Path) -> None:
        """Load model weights only from a pretrain checkpoint (either optimizer-key spelling)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"loaded base weights from {path.relative_to(ROOT)} "
            f"(pretrain step {ckpt.get('step', '?')}, val_loss {ckpt.get('best_val_loss', float('nan')):.4f})"
        )

    def _autocast(self):
        if self.cfg.precision == "fp32":
            from contextlib import nullcontext

            return nullcontext()
        return autocast_ctx(self.device)

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "step": self.step,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    # -- eval --------------------------------------------------------

    @torch.no_grad()
    def _masked_val_loss(self) -> float:
        """Mean assistant-token CE over the SFT val set (the fine-tuning objective on held-out QA)."""
        self.model.eval()
        losses = []
        for x, y in self.val_batches:
            _, loss = self.model(x, y)
            losses.append(float(loss))
        self.model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def _pretrain_val_loss(self) -> float:
        """Plain next-token CE on the frozen pretrain val set — the forgetting probe. Reads
        `last_aux_metrics['ce_loss']` (pure CE) exactly like `Trainer.evaluate`."""
        self.model.eval()
        losses = []
        for x, y in self.pretrain_val_batches:
            self.model(x, y)
            losses.append(self.model.last_aux_metrics["ce_loss"])
        self.model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def generate_samples(self) -> None:
        from llmlab.data.chat_format import EOT, encode_prompt

        eot_id = self.tokenizer.token_to_id(EOT)
        lines = []
        for prompt in self.cfg.sample_prompts:
            ids = encode_prompt(self.tokenizer, prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            out = self.model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
            gen = out[0].tolist()[len(ids):]
            if eot_id in gen:
                gen = gen[: gen.index(eot_id)]  # stop at the learned EOT
            lines.append(f"--- {prompt!r} ---\n{self.tokenizer.decode(gen)}\n")
        (self.run_dir / "samples" / f"step_{self.step:06d}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # -- logging --------------------------------------------------------

    def _log(self, record: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        wandb.log(record, step=self.step)

    def _evaluate_and_log(self, train_loss: float, lr: float, epoch: int) -> tuple[float, float]:
        val_loss = self._masked_val_loss()
        pretrain_val = self._pretrain_val_loss()
        if self.pretrain_val_loss_0 is None:
            self.pretrain_val_loss_0 = pretrain_val
        record = {
            "step": self.step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pretrain_val_loss": pretrain_val,
            "pretrain_val_ppl": math.exp(pretrain_val),
            "forgetting_delta": pretrain_val - self.pretrain_val_loss_0,
            "lr": lr,
            "mem_gb": mem_stats()["rss_mb"] / 1024,
            "elapsed_s": time.time() - self._start_time,
        }
        self._log(record)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(self.run_dir / "ckpt" / "best.pt")
        return val_loss, pretrain_val

    # -- core loop --------------------------------------------------------

    def fit(self) -> str:
        status = "completed"
        try:
            from tqdm import tqdm

            print(
                f"SFT: {len(self.train_ds)} train / {len(self.val_ds)} val examples, "
                f"{self.steps_per_epoch} steps/epoch × {self.cfg.epochs} epochs = {self.total_steps} steps"
            )
            pbar = tqdm(total=self.total_steps)
            for epoch in range(self.cfg.epochs):
                batches = self.train_ds.epoch_batches(
                    self.cfg.batch_size, self.cfg.seed, epoch, self.device
                )
                for x, y in batches:
                    lr = sft_lr_at_step(self.step, self.total_steps, self.warmup_steps, self.cfg)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr
                    self.optimizer.zero_grad(set_to_none=True)

                    self.model.train()
                    with self._autocast():
                        _, loss = self.model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                    train_loss = float(loss.detach())
                    if self.step % self.cfg.eval_every == 0:
                        val_loss, pretrain_val = self._evaluate_and_log(train_loss, lr, epoch)
                        pbar.set_postfix(
                            loss=f"{train_loss:.3f}", val=f"{val_loss:.3f}",
                            pt_ppl=f"{math.exp(pretrain_val):.1f}",
                        )
                    if self.step % self.cfg.sample_every == 0:
                        self.generate_samples()
                    self.step += 1
                    pbar.update(1)
            pbar.close()
            # Final eval + sample at the true end of training.
            self._evaluate_and_log(train_loss, lr, self.cfg.epochs - 1)
            self.generate_samples()
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
            round(len(self.train_ds) * self.cfg.epochs / 1e6, 4),  # examples seen (M), not tokens
            round(self.best_val_loss, 4) if self.best_val_loss != float("inf") else "-",
            round(math.exp(self.best_val_loss), 2) if self.best_val_loss != float("inf") else "-",
            round((time.time() - self._start_time) / 3600, 3),
            self._wandb_run.url if self._wandb_run and self.cfg.logging.wandb_mode == "online" else "-",
            f"{status} SFT {self.step}/{self.total_steps} steps -- review and fill in notes.md",
        ]
        with REGISTRY_PATH.open("a", newline="") as f:
            csv.writer(f).writerow(row)
