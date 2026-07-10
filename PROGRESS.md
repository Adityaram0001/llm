# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 1 — `docs/phases/phase1_data.md`
**Last session:** 2026-07-10 — Phase 0 completed: venv + deps installed, `llmlab` editable
install verified, MPS/bf16/SDPA/wandb all check out (`scripts/verify_env.py`), throughput
benchmarked (`scripts/bench_mps.py`, see D-008), guided notebook done.
**Open blockers:** none. The D-008 flag (hero run ≈ 1.5–3 weeks on the Mac) is resolved in
principle by **D-010**: rented RTX 5090 as burst compute for M/L-tier runs (playbook
`docs/CLOUD.md`, scripts in `scripts/cloud/`). Final go/no-go + provider choice happens when
the first big run is actually needed (phase 4 M-tier or phase 9).

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | done |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | todo |
| 2 | Tokenizers (scratch + HF) | `docs/phases/phase2_tokenizer.md` | todo |
| 3 | Model architecture | `docs/phases/phase3_architecture.md` | todo |
| 4 | Training engine + first pretrain | `docs/phases/phase4_training.md` | todo |
| 5 | Ablation lab (research techniques) | `docs/phases/phase5_ablations.md` | todo |
| 6 | Evaluation suite | `docs/phases/phase6_evaluation.md` | todo |
| 7 | Data factory (DeepSeek-assisted) | `docs/phases/phase7_data_factory.md` | todo |
| 8 | Fine-tuning: SFT / LoRA / DPO | `docs/phases/phase8_finetuning.md` | todo |
| 9 | Capstone: 100M hero run + report | `docs/phases/phase9_capstone.md` | todo |

## Phase 0 checklist (done)

- [x] `scripts/setup.sh` run: `.venv` created, requirements installed, `llmlab` editable install
- [x] `scripts/verify_env.py`: MPS available, bf16 autocast works, seed utility works
- [x] `scripts/bench_mps.py`: measured matmul TFLOPS + tokens/sec on a dummy ~9.1M-param transformer
- [x] Throughput numbers recorded in `docs/DECISIONS.md` (D-008; sets the compute budget for everything)
- [x] `notebooks/00_mps_playground.ipynb`: tensors on mps, autocast dtypes, sync timing pitfall, memory readout — executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated; phase marked done

## Run ledger (latest 10 — full list in experiments/registry.csv)

_(none yet — phase 0 was environment setup, no training runs)_

## Notes for next session

- Start with Phase 1. Read `docs/phases/phase1_data.md`.
- Environment is ready: `source .venv/bin/activate`, `llmlab` importable, jupyter kernel `llm-lab`
  registered. `src/llmlab/utils.py` has `set_seed`, `get_device`, `param_count`, `mem_stats` —
  reuse these rather than re-deriving them in phase 1+ scripts.
- Micro-batch guidance from D-008 (for whenever phase 4 needs training defaults): at seq_len 512
  the throughput plateau is around micro-batch 8-16; don't push batch size to the edge of what
  fits in MPS memory — there's a cliff (3-15x slowdown) well before a real OOM.
- D-008 timeline tension resolved by D-010 (cloud burst option). From phase 4 onward, ALL
  training code must follow `docs/CLOUD.md` portability rules (device via
  `llmlab.utils.get_device()`/`autocast_ctx()` — already updated to be cuda>mps>cpu aware).
  The user has never rented a GPU: when the first cloud run comes up, walk CLOUD.md step by
  step and suggest the $1 practice rental first.
