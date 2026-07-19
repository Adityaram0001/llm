"""DPOTrainer: direct preference optimization loop (phase 8, Part C).

Shape mirrors `SFTTrainer` (warm start, finite epochs, forgetting probe, registry row) but with
two structural differences that ARE the point of this phase:
  1. **Two models, one trainable.** `self.model` (policy) and `self.ref_model` (frozen reference)
     both start as the SAME SFT checkpoint (`sft_run`/`sft_ckpt_name` — Part A's full fine-tune by
     default), loaded via `sft_infer.load_finetuned` so a LoRA SFT source would work too. Only the
     policy gets an optimizer; the reference is `requires_grad_(False)` and always run under
     `torch.no_grad()`.
  2. **Paired batches, one scalar loss.** Each step does FOUR forward passes (policy x
     {chosen, rejected}, reference x {chosen, rejected}) via `dpo.sequence_logprobs`, then
     `dpo.dpo_loss` turns the four log-probs into one scalar loss plus reward-margin/accuracy and
     a KL-drift diagnostic — the numbers Part C's spec asks to track "instead of" a training loss
     you'd otherwise just watch fall.
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

from llmlab.data.dpo_loader import DPODataset
from llmlab.data.loader import MixedSourceLoader, Source
from llmlab.model import GPT, ModelConfig
from llmlab.utils import autocast_ctx, get_device, mem_stats, set_seed

from .dpo import dpo_loss, sequence_logprobs
from .dpo_config import DPOConfig
from .sft_infer import load_finetuned
from .sft_trainer import sft_lr_at_step  # identical warmup->cosine schedule shape
from .trainer import ROOT, REGISTRY_PATH, build_param_groups


class DPOTrainer:
    def __init__(self, cfg: DPOConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.device = torch.device(cfg.device) if cfg.device else get_device()
        set_seed(cfg.seed)

        sft_run_dir = ROOT / cfg.sft_run
        self.model, self.tokenizer, sft_cfg = load_finetuned(sft_run_dir, cfg.sft_ckpt_name, self.device)
        self.model.train()
        assert sft_cfg["model_config"] == cfg.model_config, (
            f"DPOConfig.model_config ({cfg.model_config}) must match the SFT run's own "
            f"({sft_cfg['model_config']}) -- they must be the SAME architecture."
        )

        self.ref_model, _, _ = load_finetuned(sft_run_dir, cfg.sft_ckpt_name, self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas
        )
        self.last_tok_s = 0.0

        self.train_ds = DPODataset.from_jsonl(
            ROOT / cfg.train_file, self.tokenizer, max_len=cfg.max_len, supervise_eot=cfg.supervise_eot
        )
        self.val_ds = DPODataset.from_jsonl(
            ROOT / cfg.val_file, self.tokenizer, max_len=cfg.max_len, supervise_eot=cfg.supervise_eot
        )
        self.val_batches = self.val_ds.eval_batches(cfg.batch_size, self.device)

        # Frozen pretrain-val batches for the catastrophic-forgetting probe (identical mechanic to
        # SFTTrainer -- lets Part C's forgetting number sit next to Part A/B's in the same units).
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
        self.pretrain_val_loss_0 = None
        self._start_time = time.time()

        self._wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            name=run_dir.name,
            config=cfg.to_dict(),
            mode=cfg.logging.wandb_mode,
            dir=str(run_dir),
        )
        signal.signal(signal.SIGINT, signal.default_int_handler)

    # -- autocast --------------------------------------------------------

    def _autocast(self):
        if self.cfg.precision == "fp32":
            from contextlib import nullcontext

            return nullcontext()
        return autocast_ctx(self.device)

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Only the policy is saved -- the reference is always reconstructible from
        `sft_run`/`sft_ckpt_name` (already recorded in this checkpoint), never trained."""
        torch.save({
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sft_run": self.cfg.sft_run,
            "sft_ckpt_name": self.cfg.sft_ckpt_name,
        }, path)

    # -- one paired forward/loss --------------------------------------------------------

    def _step_loss(self, x_c, y_c, x_r, y_r, grad: bool) -> tuple[torch.Tensor, dict]:
        with self._autocast():
            policy_c = sequence_logprobs(self.model, x_c, y_c)
            policy_r = sequence_logprobs(self.model, x_r, y_r)
            with torch.no_grad():
                ref_c = sequence_logprobs(self.ref_model, x_c, y_c)
                ref_r = sequence_logprobs(self.ref_model, x_r, y_r)
            loss, metrics = dpo_loss(policy_c, policy_r, ref_c, ref_r, self.cfg.beta)
        return loss, metrics

    # -- eval --------------------------------------------------------

    @torch.no_grad()
    def _val_metrics(self) -> dict:
        self.model.eval()
        losses, accs, margins, kl_c, kl_r = [], [], [], [], []
        for x_c, y_c, x_r, y_r in self.val_batches:
            loss, m = self._step_loss(x_c, y_c, x_r, y_r, grad=False)
            losses.append(float(loss))
            accs.append(m["reward_accuracy"])
            margins.append(m["reward_margin"])
            kl_c.append(m["kl_chosen"])
            kl_r.append(m["kl_rejected"])
        self.model.train()
        n = len(losses)
        return {
            "val_loss": sum(losses) / n,
            "val_reward_accuracy": sum(accs) / n,
            "val_reward_margin": sum(margins) / n,
            "val_kl_chosen": sum(kl_c) / n,
            "val_kl_rejected": sum(kl_r) / n,
        }

    @torch.no_grad()
    def _pretrain_val_loss(self) -> float:
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
                gen = gen[: gen.index(eot_id)]
            lines.append(f"--- {prompt!r} ---\n{self.tokenizer.decode(gen)}\n")
        (self.run_dir / "samples" / f"step_{self.step:06d}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # -- logging --------------------------------------------------------

    def _log(self, record: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        wandb.log(record, step=self.step)

    def _evaluate_and_log(self, train_loss: float, train_metrics: dict, lr: float, epoch: int) -> float:
        val = self._val_metrics()
        pretrain_val = self._pretrain_val_loss()
        if self.pretrain_val_loss_0 is None:
            self.pretrain_val_loss_0 = pretrain_val
        record = {
            "step": self.step,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_reward_accuracy": train_metrics["reward_accuracy"],
            "train_reward_margin": train_metrics["reward_margin"],
            "pretrain_val_loss": pretrain_val,
            "pretrain_val_ppl": math.exp(pretrain_val),
            "forgetting_delta": pretrain_val - self.pretrain_val_loss_0,
            "lr": lr,
            "tok_s": self.last_tok_s,
            "mem_gb": mem_stats()["rss_mb"] / 1024,
            "elapsed_s": time.time() - self._start_time,
            **val,
        }
        self._log(record)
        if val["val_loss"] < self.best_val_loss:
            self.best_val_loss = val["val_loss"]
            self.save_checkpoint(self.run_dir / "ckpt" / "best.pt")
        return val["val_loss"]

    # -- core loop --------------------------------------------------------

    def fit(self) -> str:
        status = "completed"
        try:
            from tqdm import tqdm

            print(
                f"DPO: {len(self.train_ds)} train / {len(self.val_ds)} val pairs, "
                f"{self.steps_per_epoch} steps/epoch x {self.cfg.epochs} epochs = {self.total_steps} steps, "
                f"beta={self.cfg.beta}"
            )
            pbar = tqdm(total=self.total_steps)
            for epoch in range(self.cfg.epochs):
                batches = self.train_ds.epoch_batches(
                    self.cfg.batch_size, self.cfg.seed, epoch, self.device
                )
                for x_c, y_c, x_r, y_r in batches:
                    lr = sft_lr_at_step(self.step, self.total_steps, self.warmup_steps, self.cfg)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr
                    self.optimizer.zero_grad(set_to_none=True)

                    t0 = time.time()
                    loss, train_metrics = self._step_loss(x_c, y_c, x_r, y_r, grad=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad), self.cfg.grad_clip
                    )
                    self.optimizer.step()
                    self.last_tok_s = (x_c.numel() + x_r.numel()) / (time.time() - t0)

                    train_loss = float(loss.detach())
                    if self.step % self.cfg.eval_every == 0:
                        val_loss = self._evaluate_and_log(train_loss, train_metrics, lr, epoch)
                        pbar.set_postfix(
                            loss=f"{train_loss:.3f}", val=f"{val_loss:.3f}",
                            acc=f"{train_metrics['reward_accuracy']:.2f}",
                            margin=f"{train_metrics['reward_margin']:.3f}",
                        )
                    if self.step % self.cfg.sample_every == 0:
                        self.generate_samples()
                    self.step += 1
                    pbar.update(1)
            pbar.close()
            self._evaluate_and_log(train_loss, train_metrics, lr, self.cfg.epochs - 1)
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
            round(len(self.train_ds) * self.cfg.epochs / 1e6, 4),
            round(self.best_val_loss, 4) if self.best_val_loss != float("inf") else "-",
            "-",  # ppl column doesn't apply to a DPO loss -- see notes.md for reward metrics
            round((time.time() - self._start_time) / 3600, 3),
            self._wandb_run.url if self._wandb_run and self.cfg.logging.wandb_mode == "online" else "-",
            f"{status} DPO {self.step}/{self.total_steps} steps -- review and fill in notes.md",
        ]
        with REGISTRY_PATH.open("a", newline="") as f:
            csv.writer(f).writerow(row)
