# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 0 — `docs/phases/phase0_setup.md`
**Last session:** 2026-07-10 — project scaffolded by planning session (Fable). No code written yet.
**Open blockers:** none

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | todo |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | todo |
| 2 | Tokenizers (scratch + HF) | `docs/phases/phase2_tokenizer.md` | todo |
| 3 | Model architecture | `docs/phases/phase3_architecture.md` | todo |
| 4 | Training engine + first pretrain | `docs/phases/phase4_training.md` | todo |
| 5 | Ablation lab (research techniques) | `docs/phases/phase5_ablations.md` | todo |
| 6 | Evaluation suite | `docs/phases/phase6_evaluation.md` | todo |
| 7 | Data factory (DeepSeek-assisted) | `docs/phases/phase7_data_factory.md` | todo |
| 8 | Fine-tuning: SFT / LoRA / DPO | `docs/phases/phase8_finetuning.md` | todo |
| 9 | Capstone: 100M hero run + report | `docs/phases/phase9_capstone.md` | todo |

## Phase 0 checklist (active)

- [ ] `scripts/setup.sh` run: `.venv` created, requirements installed, `llmlab` editable install
- [ ] `scripts/verify_env.py`: MPS available, bf16 autocast works, seed utility works
- [ ] `scripts/bench_mps.py`: measured matmul TFLOPS + tokens/sec on a dummy 10M transformer
- [ ] Throughput numbers recorded in `docs/DECISIONS.md` (they set the compute budget for everything)
- [ ] `notebooks/00_mps_playground.ipynb`: user has poked at tensors on MPS
- [ ] PROGRESS.md + DECISIONS.md updated; phase marked done

## Run ledger (latest 10 — full list in experiments/registry.csv)

_(none yet)_

## Notes for next session

- Start with Phase 0. Read `docs/phases/phase0_setup.md`.
- Environment has NOT been created yet — nothing is installed, `.venv` does not exist.
