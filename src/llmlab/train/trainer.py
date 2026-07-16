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
from .optimizers import Lion, Muon

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


def _split_params_by_ndim(model: torch.nn.Module) -> tuple[list, list]:
    """`(matrix_params, vector_params)` -- `matrix_params` are the 2D+ hidden weight matrices
    (attention/FFN projections) that get weight decay under AdamW or are Muon-eligible;
    `vector_params` are embeddings/norm gains (ndim<2, or named tok_emb/pos_emb), which get
    neither weight decay nor Muon's orthogonalization (see `build_param_groups` and
    `Trainer._build_optimizers`)."""
    matrix, vector = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name or "pos_emb" in name:
            vector.append(p)
        else:
            matrix.append(p)
    return matrix, vector


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Two AdamW param groups: matrix weights get weight decay, everything else doesn't.

    Norm gains and embeddings are excluded because weight decay's "shrink toward zero"
    regularization doesn't make sense for them: a norm gain is a single learned scale/shift, not
    a projection whose magnitude trades off against overfitting; embeddings are a lookup table
    where decaying a rarely-seen token's row toward zero actively destroys its (already
    data-starved) representation rather than regularizing it. This model has no biases
    (`bias=False` throughout), so in practice the no-decay group is exactly {tok_emb, norms}.
    """
    decay, no_decay = _split_params_by_ndim(model)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _schedule_multiplier(step: int, cfg: TrainConfig) -> float:
    """lr(step) / peak_lr, independent of which optimizer/peak-lr it's scaling -- lets the
    Muon+AdamW hybrid apply the same warmup/decay *shape* to two different peak values
    (`muon_lr`, `lr`) from one schedule definition. Three shapes (`OptimConfig.schedule`):
    - cosine: warmup -> cosine decay to `lr_min_ratio` by `max_steps` (phase 4/5's default).
    - constant: warmup -> flat at 1.0 for the rest of training (no decay at all).
    - wsd (Hu et al. '24, MiniCPM): warmup -> flat at 1.0 ("stable") -> linear decay to
      `lr_min_ratio` over the last `wsd_decay_ratio` fraction of steps. The point of WSD is that
      the "stable" phase doesn't need to know the eventual total budget -- decay can start from
      ANY checkpoint taken during it, producing a usable model at whatever budget you decide to
      stop at (see the wave_d_wsd_branch_* configs/notes for a demonstration).
    """
    o = cfg.optim
    if step < o.warmup_steps:
        return (step + 1) / o.warmup_steps
    if o.schedule == "constant":
        return 1.0
    if o.schedule == "cosine":
        if step >= cfg.max_steps:
            return o.lr_min_ratio
        progress = (step - o.warmup_steps) / max(1, cfg.max_steps - o.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * progress))
        return o.lr_min_ratio + coeff * (1 - o.lr_min_ratio)
    if o.schedule == "wsd":
        decay_start = cfg.max_steps * (1 - o.wsd_decay_ratio)
        if step < decay_start:
            return 1.0
        if step >= cfg.max_steps:
            return o.lr_min_ratio
        progress = (step - decay_start) / max(1, cfg.max_steps - decay_start)
        return 1.0 - progress * (1 - o.lr_min_ratio)
    raise ValueError(f"unknown schedule {o.schedule!r}")


def lr_at_step(step: int, cfg: TrainConfig, base_lr: float | None = None) -> float:
    """`base_lr` defaults to `cfg.optim.lr`; the Muon+AdamW hybrid calls this twice per step
    with `base_lr=cfg.optim.muon_lr` and `base_lr=cfg.optim.lr` to schedule both optimizers off
    the same warmup/decay shape (see `_schedule_multiplier`)."""
    base = cfg.optim.lr if base_lr is None else base_lr
    return base * _schedule_multiplier(step, cfg)


class Trainer:
    def __init__(self, cfg: TrainConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.device = torch.device(cfg.device) if cfg.device else get_device()
        set_seed(cfg.seed)

        # D-043: micro_batch=16 is the Mac-tuned plateau (D-022, MPS-specific) -- on a rented
        # CUDA GPU it leaves real throughput on the table (5090 S-tier sweet spot is mb=64, ~4x
        # more tok/s per docs/CLOUD_GPUHUB.md's sweep). Waves A-C's configs missed this despite
        # it already being documented; a runtime nag is harder to miss than a doc. Not a hard
        # error -- some ablations (e.g. Wave E) deliberately vary micro_batch, including =16.
        if self.device.type == "cuda" and cfg.batch.micro_batch <= 16:
            print(
                f"WARNING: micro_batch={cfg.batch.micro_batch} on cuda -- this is the Mac/MPS-"
                "tuned default, not the cloud sweet spot. See docs/CLOUD_GPUHUB.md section 10 "
                "(S-tier RTX 5090 sweet spot is mb=64) before assuming this run is at full "
                "throughput. Ignore if micro_batch is the deliberate ablation variable."
            )

        model_cfg = ModelConfig.from_yaml(str(ROOT / cfg.model_config))
        self.model = GPT(model_cfg).to(self.device)
        self.model.gradient_checkpointing = cfg.gradient_checkpointing
        self._raw_model = self.model  # always the uncompiled module; used for checkpointing
        self.optimizers, self._base_lrs = self._build_optimizers()

        # Wave E: torch.compile is a graph-capture optimization, not a training-math change --
        # attempted here so a failure surfaces at startup rather than mid-run, and logged rather
        # than silently falling back (CLAUDE.md: torch.compile on MPS is unreliable, treat as an
        # optional experiment). `self.model` becomes the compiled wrapper for forward/backward;
        # `self._raw_model` still points at the original module so checkpoint state_dict keys
        # never depend on torch.compile's (version-dependent) attribute-naming internals.
        self.compile_status = "disabled"
        if cfg.compile:
            try:
                self.model = torch.compile(self.model)
                self.compile_status = "enabled"
            except Exception as e:  # pragma: no cover -- environment-dependent compile failures
                self.compile_status = f"failed: {e}"
                print(f"torch.compile failed, continuing uncompiled: {e}")

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

    # -- optimizer construction --------------------------------------------------------

    def _build_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[float]]:
        """Returns `(optimizers, base_lrs)`, kept parallel: `train_step` schedules
        `optimizers[i]`'s lr off `base_lrs[i]` every step (see `lr_at_step`). `adamw`/`lion`
        are a single optimizer over both param groups (matrix weights get `weight_decay`,
        vectors don't -- see `build_param_groups`); `muon` is the nanoGPT-speedrun hybrid: Muon
        orthogonalizes the 2D hidden matrices, a plain (no-decay) AdamW handles everything else
        (embeddings, norm gains) since Muon's "orthogonalize a matrix update" framing doesn't
        apply to those.
        """
        o = self.cfg.optim
        if o.optimizer == "adamw":
            opt = torch.optim.AdamW(
                build_param_groups(self.model, o.weight_decay), lr=o.lr, betas=o.betas
            )
            return [opt], [o.lr]
        if o.optimizer == "lion":
            opt = Lion(build_param_groups(self.model, o.weight_decay), lr=o.lr, betas=o.betas)
            return [opt], [o.lr]
        if o.optimizer == "muon":
            matrix_params, vector_params = _split_params_by_ndim(self.model)
            muon_opt = Muon(
                matrix_params, lr=o.muon_lr, momentum=o.muon_momentum, ns_steps=o.muon_ns_steps
            )
            adamw_opt = torch.optim.AdamW(
                [{"params": vector_params, "weight_decay": 0.0}], lr=o.lr, betas=o.betas
            )
            return [muon_opt, adamw_opt], [o.muon_lr, o.lr]
        raise ValueError(f"unknown optimizer {o.optimizer!r}")

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self._raw_model.state_dict(),
                "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        for opt, state in zip(self.optimizers, ckpt["optimizer_state_dicts"]):
            opt.load_state_dict(state)
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val_loss = ckpt["best_val_loss"]

    # -- core loop --------------------------------------------------------

    def _autocast(self):
        """Wave E precision knob: `precision="bf16"` (default) is `autocast_ctx`'s mixed
        precision; `precision="fp32"` disables autocast entirely (a plain `nullcontext`), so
        every matmul actually runs in fp32 rather than merely widening the accumulate dtype."""
        if self.cfg.precision == "fp32":
            from contextlib import nullcontext

            return nullcontext()
        if self.cfg.precision == "bf16":
            return autocast_ctx(self.device)
        raise ValueError(f"unknown precision {self.cfg.precision!r}")

    def train_step(self) -> tuple[float, float, float, dict]:
        self.model.train()
        lrs = [lr_at_step(self.step, self.cfg, base_lr=b) for b in self._base_lrs]
        for opt, lr in zip(self.optimizers, lrs):
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

        z_loss_weight = self.cfg.optim.z_loss_weight
        total_loss = 0.0
        aux_metrics: dict = {}
        for micro in range(self.cfg.batch.grad_accum):
            data_step = self.step * self.cfg.batch.grad_accum + micro
            x, y = self.train_loader.get_batch(data_step, self.cfg.batch.micro_batch, self.device)
            with self._autocast():
                logits, loss = self.model(x, y)
                if z_loss_weight:
                    # PaLM '22 z-loss: penalize log Z (the softmax normalizer) growing large,
                    # which otherwise drifts unbounded since only *differences* between logits
                    # matter to cross-entropy -- a stability aid, not an accuracy one.
                    z_loss = logits.logsumexp(dim=-1).pow(2).mean()
                    loss = loss + z_loss_weight * z_loss
            # Wave F: grabbed from the LAST micro-batch only (diagnostic logging, not part of
            # the optimized objective -- that already correctly sums moe/mtp loss terms into
            # `loss` above, per micro-batch, via loss.backward()).
            aux_metrics = self._raw_model.last_aux_metrics
            loss = loss / self.cfg.batch.grad_accum
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optim.grad_clip
        )
        # Wave F: DeepSeek-V3's aux-loss-free bias update is gradient-free bookkeeping, not part
        # of backward/optimizer.step() -- fires once per optimizer step, aggregating load across
        # every micro-batch this step just ran (see MoEFFN.update_bias).
        moe_cfg = self._raw_model.cfg.moe
        if moe_cfg is not None:
            self._raw_model.update_moe_bias(moe_cfg.bias_update_rate)
        for opt in self.optimizers:
            opt.step()
        self.tokens_seen += self.tokens_per_step
        return total_loss, float(grad_norm), lrs[0], aux_metrics

    @torch.no_grad()
    def evaluate(self) -> float:
        """Deliberately NOT gated by `precision` -- eval always ran in plain fp32 (no autocast)
        even before Wave E's precision knob existed, so every wave's val_loss stays measured
        the same way regardless of what precision a given run trained under.

        Wave F (phase 5): reads `last_aux_metrics["ce_loss"]` -- pure next-token cross-entropy,
        NOT `forward()`'s returned `loss` (which for moe/mtp configs also carries the weighted
        aux/balance terms that `train_step` needs for backprop). val_loss must stay directly
        comparable to every other wave's noise-floor convention (docs/EXPERIMENTS.md), which
        only ever measured plain CE.
        """
        self.model.eval()
        losses = []
        for x, y in self._eval_batches:
            self.model(x, y)
            losses.append(self._raw_model.last_aux_metrics["ce_loss"])
        self.model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def generate_samples(self) -> None:
        lines = []
        for prompt in self.cfg.sampling.prompts:
            ids = self.tokenizer.encode(prompt).ids
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            out = self._raw_model.generate(
                idx, max_new_tokens=self.cfg.sampling.max_new_tokens, temperature=0.8, top_k=40
            )
            lines.append(f"--- prompt: {prompt!r} ---\n{self.tokenizer.decode(out[0].tolist())}\n")
        out_path = self.run_dir / "samples" / f"step_{self.step:06d}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def _log(
        self,
        step: int,
        train_loss: float,
        grad_norm: float,
        lr: float,
        tokens_per_sec: float,
        val_loss: float | None,
        aux_metrics: dict | None = None,
    ) -> None:
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
        if aux_metrics:  # Wave F: moe_aux_loss / expert_load (per-layer) / mtp_loss, if present
            record.update(aux_metrics)
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
                train_loss, grad_norm, lr, aux_metrics = self.train_step()
                step_time = time.time() - t0
                tokens_per_sec = self.tokens_per_step / step_time

                val_loss = None
                if step % self.cfg.eval.eval_every == 0:
                    val_loss = self.evaluate()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(self.run_dir / "ckpt" / "best.pt")

                if step % self.cfg.logging.log_every == 0 or val_loss is not None:
                    self._log(step, train_loss, grad_norm, lr, tokens_per_sec, val_loss, aux_metrics)
                    pbar.set_postfix(loss=f"{train_loss:.3f}", val=f"{val_loss:.3f}" if val_loss else "-", lr=f"{lr:.2e}")

                if step % self.cfg.sampling.sample_every == 0:
                    self.generate_samples()

                self.step = step + 1  # `step` is now fully done -- safe resume point
                if self.step % self.cfg.checkpoint_every == 0:
                    self.save_checkpoint(self.run_dir / "ckpt" / "latest.pt")
                if self.step in self.cfg.milestone_steps:
                    self.save_checkpoint(self.run_dir / "ckpt" / f"step_{self.step:06d}.pt")
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
            round(self._raw_model.num_params() / 1e6, 2),
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
