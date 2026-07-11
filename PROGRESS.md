# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 3 — `docs/phases/phase3_architecture.md`
**Last session:** 2026-07-10 — Phase 2 completed: Part A from-scratch byte-level BPE
(`src/llmlab/tokenizer/bpe_scratch.py`, `notebooks/02_bpe_from_scratch.ipynb`, trained on one
book). Part B HF byte-level BPE tokenizers at 8k/16k/32k trained on the full S-tier corpus
(`src/llmlab/tokenizer/train_hf.py`, saved to `data/tokenized/tokenizers/hf_bpe_{8k,16k,32k}/`).
Part C comparison (`notebooks/03_tokenizer_compare.ipynb`) measured fertility, compression,
vocab utilization, rare-word splitting, and embedding-table cost for 8k/16k/32k + GPT-2's
50k + the scratch tokenizer; **chose HF BPE 16k vocab** (D-014) — captures most of 32k's
fertility/rare-word gains at half the embedding-table cost, and avoids 32k's utilization
problem (only 49.3% of its vocab fires on our own held-out text). Part D tokenized the full
corpus with the chosen tokenizer via `scripts/tokenize_corpus.py`:
`data/tokenized/hf_bpe_16k/{train,val}.bin` (uint16 memmap) — 17,665,275 train tokens / 111
docs, 179,655 val tokens / 3 docs. Registered as 5 comparison rows in
`experiments/registry.csv` (p2, no real training involved — tier/params/tokens/loss columns
are "-").
**Open blockers:** none. The D-008 flag (hero run ≈ 1.5–3 weeks on the Mac) is resolved in
principle by **D-010**: rented RTX 5090 as burst compute for M/L-tier runs (playbook
`docs/CLOUD.md`, scripts in `scripts/cloud/`). Final go/no-go + provider choice happens when
the first big run is actually needed (phase 4 M-tier or phase 9).

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | done |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | done |
| 2 | Tokenizers (scratch + HF) | `docs/phases/phase2_tokenizer.md` | done |
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

## Phase 2 checklist (done)

- [x] `src/llmlab/tokenizer/bpe_scratch.py`: pure-Python byte-level BPE (train/encode/decode),
  supports `pretok_mode` in {none, whitespace, gpt2}
- [x] `notebooks/02_bpe_from_scratch.ipynb`: trained on `marcus-aurelius-meditations.txt`,
  shows first merges, pretokenization comparison, vocab-size-vs-compression curve, byte-level
  no-OOV demo (emoji/CJK/tags round-trip) — executes cleanly end to end
- [x] `src/llmlab/tokenizer/train_hf.py`: HF `ByteLevelBPETokenizer` trained on the full
  S-tier corpus at 8k/16k/32k, saved to `data/tokenized/tokenizers/hf_bpe_{8k,16k,32k}/`
  (`tokenizer.json` + `vocab.json`/`merges.txt`); special tokens `<|endoftext|>`, `<|pad|>`,
  `<|user|>`, `<|assistant|>` reserved for phase 8
- [x] `notebooks/03_tokenizer_compare.ipynb`: fertility/compression, vocab utilization,
  rare-word splitting, numbers/punctuation, embedding-table cost math, for scratch-bpe-8k /
  hf-bpe-8k/16k/32k / gpt2-50k — figures + written verdict — executes cleanly end to end
- [x] Decision logged: **D-014, HF BPE 16k vocab** chosen (user reviewed the comparison
  table); 5 comparison rows registered in `experiments/registry.csv` (p2, non-training rows)
- [x] `scripts/tokenize_corpus.py`: encodes train+val corpus → `data/tokenized/hf_bpe_16k/
  {train,val}.bin` (uint16 memmap) + `meta.json` (vocab size, per-doc token offsets, token
  counts); verified via decoding random slices. 17,665,275 train tokens (111 docs), 179,655
  val tokens (3 docs)
- [x] PROGRESS.md + DECISIONS.md updated (D-014); phase marked done

## Rework queue (see CLAUDE.md "Change management")

| ID | What | Why | Fix in phase | Status |
|----|------|-----|--------------|--------|
| RW-1 | Tokenize TinyStories supplement (+any added corpus) with hf_bpe_16k → `data/tokenized/hf_bpe_16k/supplement_*.bin`; extend `scripts/tokenize_corpus.py` | Supplement was left raw-only in phase 2 by design; needed once M/L-tier runs mix it in. Final size depends on the phase-3 tier/data-budget decision | 4 (before first M-tier run) | todo |
| RW-2 | If phase 3 grows L-tier beyond ~105M: recompute D-008 wall-clock extrapolations + D-010 cloud cost estimate; check corpus covers ≥20 tok/param or log the deliberate shortfall | Tier sizes set pre-tokenizer (D-001) are being finalized vocab-aware in phase 3 | 3 | todo |

## Run ledger (latest 10 — full list in experiments/registry.csv)

5 non-training comparison rows from the phase-2 tokenizer study (`20260710_p2_tokenizer-*`) —
see `experiments/registry.csv`. No model training has happened yet (phases 0-2 were
environment + data + tokenizer setup).

## Notes for next session

- Start with Phase 3. Read `docs/phases/phase3_architecture.md`. Its decision points now
  include **finalizing tier parameter counts vocab-aware** (user explicitly wants the
  parameter-allocation reasoning surfaced — see `docs/learnings/20260711_parameter-allocation.md`)
  and the coupled data-budget check (RW-2).
- Tokenizer is decided (D-014): **HF BPE, 16k vocab**, files at
  `data/tokenized/tokenizers/hf_bpe_16k/` (tokenizer itself) and
  `data/tokenized/hf_bpe_16k/{train,val}.bin` + `meta.json` (tokenized corpus, uint16 memmap,
  ready for a phase-4 DataLoader). `<|endoftext|>` id is in `meta.json`'s `eot_id` field
  (0 for this tokenizer); `<|pad|>`/`<|user|>`/`<|assistant|>` ids are already reserved in the
  vocab for phase 8 — check `data/tokenized/tokenizers/hf_bpe_16k/vocab.json` if their exact
  IDs are needed.
- Corpus is ready at `data/clean/`: `books/*.txt` (110 train + 2 val in `val/books/`),
  `dictionary_prose.txt` + `dictionary.jsonl` (+ val versions), `supplement/tinystories.txt`.
  `data/clean/manifest.json` has per-file stats/sha256/license. Re-run
  `python scripts/build_corpus.py` any time to rebuild from scratch (idempotent, cached in
  `data/raw/`); add `--force` to re-download, or `--skip-books`/`--skip-dictionary`/
  `--skip-supplement` to build a subset — partial runs merge into the existing
  `data/clean/manifest.json` rather than overwriting it.
- Token budget: 17,665,275 train + 179,655 val tokens tokenized at 16k vocab (books+dictionary,
  the S-tier ablation corpus per D-006) + a much larger TinyStories supplement (~475-533M
  tokens estimated, NOT yet tokenized — stored as raw text only, not mixed into the default
  training stream — see `configs/corpus.yaml` `supplement.tinystories`) for M/L-tier runs.
  If a later phase mixes in the supplement, it will need its own tokenize_corpus.py run (the
  script currently only tokenizes books+dictionary by design, see phase 2 notes in D-014).
- Environment is ready: `source .venv/bin/activate`, `llmlab` importable, jupyter kernel `llm-lab`
  registered. `src/llmlab/utils.py` has `set_seed`, `get_device`, `param_count`, `mem_stats` —
  reuse these rather than re-deriving them in phase 3+ scripts.
- Micro-batch guidance from D-008 (for whenever phase 4 needs training defaults): at seq_len 512
  the throughput plateau is around micro-batch 8-16; don't push batch size to the edge of what
  fits in MPS memory — there's a cliff (3-15x slowdown) well before a real OOM.
- D-008 timeline tension resolved by D-010 (cloud burst option). From phase 4 onward, ALL
  training code must follow `docs/CLOUD.md` portability rules (device via
  `llmlab.utils.get_device()`/`autocast_ctx()` — already updated to be cuda>mps>cpu aware).
  The user has never rented a GPU: when the first cloud run comes up, walk CLOUD.md step by
  step and suggest the $1 practice rental first.
