# Phase 4 — Training engine + first real pretrains

**Goal:** a robust, resumable, instrumented training loop; wandb + local tracking wired; the
first S-tier pretrains on books+dictionary; the run-comparison notebook. After this phase the
ablation lab is just "new config, `python scripts/train.py`".
**Effort:** 2–3 sessions (engine / first runs / comparison tooling).

## Deliverables

0. **Portability requirement (D-010):** the whole engine must run unchanged on MPS (local)
   and CUDA (rented pod) — follow `docs/CLOUD.md` rules: `get_device()`/`autocast_ctx()` from
   `llmlab.utils`, DataLoader `num_workers`/`pin_memory` as config keys, `map_location` on
   load, TF32 + optional `compile: true` config flag for CUDA. Acceptance: `--device cpu`
   smoke test passes (that's the portability canary without renting anything).
1. **`src/llmlab/data/loader.py`** — memmap dataset: random-offset contiguous slices of
   `train.bin` → `(x, y)` next-token pairs; deterministic given seed+step (needed for exact
   ablation comparability); optional doc-boundary-respecting mode (default: plain
   concat-and-chunk like GPT-2 — explain the trade-off).
2. **`src/llmlab/train/trainer.py`** — the heart. Features (each is a lesson, implement in this
   order, commit-by-commit):
   - config-driven (`configs/train_*.yaml`: lr, warmup, schedule, betas, wd, grad_clip,
     micro_batch, grad_accum, max_steps/max_tokens, eval_every, sample_every, seed, run_id…)
   - AdamW with **param groups**: no weight decay on norms/biases/embeddings (explain why)
   - LR schedule: linear warmup → cosine to lr_min; pluggable (WSD arrives in P5-D)
   - bf16 autocast on mps; grad accumulation; grad-norm clipping (log the norm!)
   - eval loop on `val.bin` (fixed batches, model.eval, no_grad)
   - text sampling every N steps → `samples/step_%06d.txt` (fixed prompts incl. a dictionary
     prompt "ephemeral (adjective):" — watching these evolve is the best part)
   - checkpointing: `latest.pt` (model+optimizer+scheduler+step+rng states) — **resume must be
     bit-exact-ish**; `best.pt` on val loss
   - logging: tokens/sec, RSS, mps memory, grad_norm, lr → `metrics.jsonl` + wandb; tqdm console
   - graceful Ctrl-C (save latest, write registry row)
3. **`scripts/train.py`** — CLI: `python scripts/train.py --config configs/train_s_baseline.yaml
   [--resume experiments/<run_id>]`. Auto-creates run folder, dumps resolved config, appends
   registry row at end.
4. **First experiments (register all):**
   - `p4_smoke`: S-tier, 10 min run — everything works end to end
   - `p4_s_baseline`: S-tier, ~100M tokens on books+dictionary (~1–2h). THE reference run.
   - `p4_s_lr_sweep`: 3 short runs at lr ×0.3 / ×1 / ×3 — first real hyperparameter lesson;
     watch divergence happen on purpose.
   - resume test: kill baseline mid-run, resume, verify curve continuity.
5. **`notebooks/05_compare_runs.ipynb`** (per EXPERIMENTS.md): loads registry + metrics.jsonl,
   plots loss-vs-tokens / loss-vs-wallclock / tok/s / memory; reusable `plot_runs([...run_ids])`.
   Also: a "read the loss curve" teaching section (initial cliff, power-law region, LR-decay dip;
   spikes and what causes them).

## Decision points

- Baseline hyperparameters for S-tier (propose: lr 3e-4ish scaled, warmup 1–2% steps, cosine to
  10%, wd 0.1, β=(0.9,0.95), clip 1.0, seq 512, effective batch ~0.25–0.5M tokens — justify
  each number vs GPT-2/nanoGPT/Chinchilla practice, then log).
- Eval cadence & val set size (compute cost of eval vs curve resolution).
- wandb run naming/grouping convention.

## Learning checkpoints

- What exactly is in a checkpoint and why optimizer state doubles its size for AdamW.
- Why effective batch size (not micro-batch) is the real hyperparameter; grad-accum math.
- How to read grad_norm; what a loss spike means; why warmup exists.
- Perplexity = exp(loss) — and why comparing ppl across different tokenizers is invalid.

## Exit criteria
Baseline S run finished & registered; samples show English-ish text; resume verified;
comparison notebook renders; milestone M1 declared in PROGRESS; decisions logged.
