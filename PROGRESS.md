# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 2 — `docs/phases/phase2_tokenizer.md`
**Last session:** 2026-07-10 — Phase 1 completed: 112-book philosophy/classics corpus (user's
20 seed authors + ~90 more auto-selected from Gutenberg's catalog metadata, see D-011) +
GCIDE dictionary (119,984 entries, D-012) + TinyStories supplement (D-013) built via
`scripts/build_corpus.py`. ~14.9M book tokens + ~2.9M dictionary tokens + ~475-533M supplement
tokens (chars/4 vs GPT-2-calibrated estimates — see `notebooks/01_corpus_stats.ipynb`).
Held-out val split by document: 2 books (Boethius' *Consolation*, Epictetus' *Enchiridion*) +
2% of dictionary entries.
**Open blockers:** none. The D-008 flag (hero run ≈ 1.5–3 weeks on the Mac) is resolved in
principle by **D-010**: rented RTX 5090 as burst compute for M/L-tier runs (playbook
`docs/CLOUD.md`, scripts in `scripts/cloud/`). Final go/no-go + provider choice happens when
the first big run is actually needed (phase 4 M-tier or phase 9).

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | done |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | done |
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

## Phase 1 checklist (done)

- [x] `configs/corpus.yaml`: 112 books (20 user-picked authors + ~90 auto-selected from
  Gutenberg's catalog metadata, see D-011), GCIDE dictionary config, TinyStories supplement flag
- [x] `src/llmlab/data/acquire.py` + `scripts/build_corpus.py`: idempotent download → clean →
  dedup → stats pipeline (downloads cached in `data/raw/`, safe to re-run)
- [x] Gutenberg boilerplate stripped, unicode NFC-normalized, whitespace collapsed,
  exact-duplicate paragraphs deduped (hash-based) — all books clean in `data/clean/books/`
- [x] GCIDE dictionary parsed (119,984 entries) into `data/clean/dictionary_prose.txt`
  (bold-term template) + `data/clean/dictionary.jsonl` (structured, for phase 6/7 eval probes)
- [x] TinyStories supplement streamed to `data/clean/supplement/tinystories.txt` (D-013)
- [x] Held-out val split by whole document: `data/clean/val/books/{boethius,epictetus}...txt` +
  2% of dictionary entries in `data/clean/val/dictionary.jsonl` — never seen in training
- [x] `data/clean/manifest.json`: source URL, license, sha256, char/word counts per file
- [x] `notebooks/01_corpus_stats.ipynb`: composition, chars/4 vs GPT-2-calibrated token
  estimates, length histogram, common-words sanity check — executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated (D-011, D-012, D-013); phase marked done

## Run ledger (latest 10 — full list in experiments/registry.csv)

_(none yet — phases 0-1 were environment + data setup, no training runs)_

## Notes for next session

- Start with Phase 2. Read `docs/phases/phase2_tokenizer.md`.
- Corpus is ready at `data/clean/`: `books/*.txt` (110 train + 2 val in `val/books/`),
  `dictionary_prose.txt` + `dictionary.jsonl` (+ val versions), `supplement/tinystories.txt`.
  `data/clean/manifest.json` has per-file stats/sha256/license. Re-run
  `python scripts/build_corpus.py` any time to rebuild from scratch (idempotent, cached in
  `data/raw/`); add `--force` to re-download, or `--skip-books`/`--skip-dictionary`/
  `--skip-supplement` to build a subset — partial runs merge into the existing
  `data/clean/manifest.json` rather than overwriting it.
- Token budget so far (see `notebooks/01_corpus_stats.ipynb` for the calibrated numbers):
  ~14.9M book tokens + ~2.9M dictionary tokens (books+dictionary is the S-tier ablation corpus,
  per D-006) + a much larger TinyStories supplement (~475-533M tokens, stored separately, not
  mixed into the default training stream — see `configs/corpus.yaml` `supplement.tinystories`)
  for M/L-tier runs.
- Environment is ready: `source .venv/bin/activate`, `llmlab` importable, jupyter kernel `llm-lab`
  registered. `src/llmlab/utils.py` has `set_seed`, `get_device`, `param_count`, `mem_stats` —
  reuse these rather than re-deriving them in phase 2+ scripts.
- Micro-batch guidance from D-008 (for whenever phase 4 needs training defaults): at seq_len 512
  the throughput plateau is around micro-batch 8-16; don't push batch size to the edge of what
  fits in MPS memory — there's a cliff (3-15x slowdown) well before a real OOM.
- D-008 timeline tension resolved by D-010 (cloud burst option). From phase 4 onward, ALL
  training code must follow `docs/CLOUD.md` portability rules (device via
  `llmlab.utils.get_device()`/`autocast_ctx()` — already updated to be cuda>mps>cpu aware).
  The user has never rented a GPU: when the first cloud run comes up, walk CLOUD.md step by
  step and suggest the $1 practice rental first.
