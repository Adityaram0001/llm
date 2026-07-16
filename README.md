# LLM-Lab — build & study a 100M-param LLM on a MacBook (+ cloud burst)

A hands-on learning project: pretrain a GPT-style language model from scratch on public-domain
books + an English dictionary, on Apple Silicon (M4, 16GB, PyTorch MPS) — then use it as a
laboratory to implement and compare training techniques from research papers (RoPE, RMSNorm,
SwiGLU, GQA, **DeepSeek's MLA / MoE / Multi-Token Prediction**, Muon, WSD schedules, LoRA,
DPO, …), with every decision logged and every run registered for comparison.

**Learning is the product. The model is the by-product.**

## Status (see `PROGRESS.md` for the live version)

Phases 0–4 are **done**: environment, corpus (books + GCIDE dictionary + TinyStories/FineWeb-Edu
supplements), tokenizer (HF BPE, 16k vocab), model architecture (`GPT` — RoPE/RMSNorm/SwiGLU/GQA,
S/M/L tiers), and the full training engine, with a real S-tier baseline run registered
(val_loss 3.50, ppl 33.2). A local Mac (MPS) + rented-GPU (gpuhub, RTX 5090 default) two-target
training path is validated end-to-end. Phase 5 (ablation lab — the project's research core) is
next.

## Map of the repo

| Path | What |
|------|------|
| `CLAUDE.md` | Rules for AI coding sessions (session protocol, hardware rules) — read first |
| `PROGRESS.md` | Live project state: active phase, checklists, rework queue, blockers |
| `docs/ROADMAP.md` | Master plan: phases 0–9, milestones, compute reality |
| `docs/TECHNIQUES.md` | Research-paper catalog with priorities & MPS feasibility |
| `docs/DECISIONS.md` | Append-only decision log (the "why" trail, D-001...) |
| `docs/EXPERIMENTS.md` | Run/ablation protocol + registry schema |
| `docs/CLOUD.md` | Rented-GPU playbook (provider-agnostic) + Mac↔Linux portability rules |
| `docs/CLOUD_GPUHUB.md` | gpuhub-specific playbook: setup script, image workflow, measured GPU capacity/pricing tables |
| `docs/WANDB.md` | wandb dashboard setup: credentials, online-vs-offline per run, syncing a pod's offline runs, how it relates to R2 |
| `docs/results/` | Ablation figures, `cloud_gpu_benchmarks.csv` (raw GPU sweep data), `recipe.md` (phase 9 input, once phase 5 lands) |
| `docs/learnings/` | Discussion-session notes (dated, indexed in `INDEX.md`) — the "what clicked" record, separate from decisions |
| `docs/phases/phaseN_*.md` | Self-contained spec for each phase (one spec ≈ one AI chat) |
| `src/llmlab/` | The python package: `model/` (config, norms, positional, attention, ffn, block, gpt), `data/` (acquire, loader), `tokenizer/` (scratch BPE, HF BPE training), `train/` (config, trainer), `utils.py` (device/seed/mem helpers) |
| `scripts/` | CLI entry points: `setup.sh`, `verify_env.py`, `bench_mps.py`, `build_corpus.py`, `tokenize_corpus.py`, `train.py`, `find_batch_size.py`, `orchestrate_p4_lr_sweep_and_baseline.py`, `cloud/` (gpuhub setup + sync scripts) |
| `configs/` | YAML configs — every run is fully described by one (`corpus.yaml`, `model_{s,m,l}.yaml`, `train_s_*.yaml`) |
| `notebooks/` | Numbered teaching/exploration notebooks 00–05 (never for real training) |
| `experiments/` | One folder per run (config + metrics.jsonl + notes.md + checkpoints) + `registry.csv` (append-only lab record) |
| `checkpoints/` | Scratch checkpoint output (gitignored; real checkpoints live inside each `experiments/<run_id>/`) |
| `data/` | raw → clean → tokenized corpus + sft datasets (gitignored) |
| `docker/` | `Dockerfile`/`entrypoint.sh` for the (currently unused — gpuhub can't pull Docker Hub images, see `docs/CLOUD_GPUHUB.md`) container path; kept documented-but-unbuilt per D-027 |
| `tools/data_factory/` | Human-in-the-loop DeepSeek dataset generator (`inbox`/`outbox`/`parsed`/`failed` batch folders) — phase 7 |
| `tests/` | `test_model.py`, `test_loader.py`, `test_trainer.py` |
| `additionals/` | Background material about the user (not project code) |

## Quick start

```bash
./scripts/setup.sh          # venv + deps + editable install + jupyter kernel
source .venv/bin/activate
python scripts/verify_env.py
```

Train locally (Mac/MPS):
```bash
python scripts/train.py --config configs/train_s_baseline.yaml
```

Burst to a rented GPU (gpuhub, RTX 5090 default — see `docs/CLOUD_GPUHUB.md` for the full
playbook, credentials in `scripts/cloud/remote.env`):
```bash
python scripts/find_batch_size.py --config configs/model_s.yaml   # calibrate micro_batch first
./scripts/cloud/sync_up.sh && ./scripts/cloud/sync_down.sh         # data/checkpoint sync
python scripts/train.py --config configs/train_s_baseline.yaml --wandb-online  # live dashboard
./scripts/cloud/push_checkpoints.sh && ./scripts/cloud/wandb_sync.sh           # archive + sync
```
Training logs to wandb (offline by default, `--wandb-online` streams live) plus local
`metrics.jsonl` always — see `docs/WANDB.md`. Checkpoints archive to R2 — see D-041.

## Working model

Each phase is executed in a fresh Claude (Sonnet) chat to conserve usage:
open chat → "Continue the LLM-Lab project, read CLAUDE.md and PROGRESS.md" → work the active
phase spec → session updates PROGRESS.md/DECISIONS.md before ending. Long training runs happen
in a plain terminal (`caffeinate -is python scripts/train.py --config ...`) with other apps closed,
or on a rented GPU per `docs/CLOUD_GPUHUB.md`.

Separate **discussion sessions** (open with "Discussion session: <topic>") are for questions/
concepts only — no code or spec changes; each ends with a note in `docs/learnings/`.
