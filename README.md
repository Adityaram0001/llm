# LLM-Lab — build & study a 100M-param LLM on a MacBook

A hands-on learning project: pretrain a GPT-style language model from scratch on public-domain
books + an English dictionary, on Apple Silicon (M4, 16GB, PyTorch MPS) — then use it as a
laboratory to implement and compare training techniques from research papers (RoPE, RMSNorm,
SwiGLU, GQA, **DeepSeek's MLA / MoE / Multi-Token Prediction**, Muon, WSD schedules, LoRA,
DPO, …), with every decision logged and every run registered for comparison.

**Learning is the product. The model is the by-product.**

## Map of the repo

| Path | What |
|------|------|
| `CLAUDE.md` | Rules for AI coding sessions (session protocol, hardware rules) — read first |
| `PROGRESS.md` | Live project state: active phase, checklists, blockers |
| `docs/ROADMAP.md` | Master plan: phases 0–9, milestones, compute reality |
| `docs/TECHNIQUES.md` | Research-paper catalog with priorities & MPS feasibility |
| `docs/DECISIONS.md` | Append-only decision log (the "why" trail) |
| `docs/EXPERIMENTS.md` | Run/ablation protocol + registry schema |
| `docs/CLOUD.md` | Rented-GPU playbook (RunPod/RTX 5090) + Mac↔Linux portability rules |
| `docs/phases/phaseN_*.md` | Self-contained spec for each phase (one spec ≈ one AI chat) |
| `src/llmlab/` | The python package: model, data, tokenizer, train, eval |
| `scripts/` | CLI entry points (train, evaluate, tokenize, bench, chat…) |
| `configs/` | YAML configs — every run is fully described by one |
| `notebooks/` | Numbered teaching/exploration notebooks (never for real training) |
| `experiments/` | One folder per run + `registry.csv` (append-only lab record) |
| `data/` | raw → clean → tokenized corpus + sft datasets (gitignored) |
| `tools/data_factory/` | Human-in-the-loop DeepSeek dataset generator |
| `additionals/` | Background material about the user (not project code) |

## Quick start

```bash
./scripts/setup.sh          # venv + deps + editable install + jupyter kernel
source .venv/bin/activate
python scripts/verify_env.py
```

## Working model

Each phase is executed in a fresh Claude (Sonnet) chat to conserve usage:
open chat → "Continue the LLM-Lab project, read CLAUDE.md and PROGRESS.md" → work the active
phase spec → session updates PROGRESS.md/DECISIONS.md before ending. Long training runs happen
in a plain terminal (`caffeinate -is python scripts/train.py --config ...`) with other apps closed.
