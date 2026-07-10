# Phase 1 — Corpus: books + dictionary (+ optional supplement)

**Goal:** a clean, deduplicated, licensed text corpus in `data/clean/`, fully characterized
(size, token estimates, composition), with a repeatable build script.
**Effort:** one session (+ download time).

## Sources (per D-003 / D-006)

1. **Books — Project Gutenberg** (public domain). USER picks 10–30 books they like (mirrors
   variety: fiction, non-fiction, essays). Download via `gutenbergpy` or plain `requests` on
   the `.txt` URLs. **Must** strip the Gutenberg header/footer boilerplate (regex on
   `*** START OF ... ***` / `*** END OF ... ***`).
2. **Dictionary — Webster's 1913 (GCIDE)**: download GCIDE, parse entries into two renderings:
   - `dictionary_prose.txt`: "**ephemeral** (adjective): lasting a very short time…" — natural
     text for pretraining.
   - `dictionary.jsonl`: `{"word":…, "pos":…, "definitions":[…]}` — structured, for eval probes
     and data-factory seeds later.
   (Alternative source if GCIDE parsing is painful: kaikki.org Wiktionary JSON extract —
   already structured; filter English, common words.)
3. **Optional supplement (for M/L-tier runs)** — ASK USER before downloading (~1–2GB):
   TinyStories (HF `roneneldan/TinyStories`) and/or a FineWeb-Edu sample. Store separately:
   `data/clean/supplement/`.

## Deliverables

1. **`src/llmlab/data/acquire.py`** + **`scripts/build_corpus.py`**: idempotent end-to-end build
   (download → clean → dedup → stats). Book list lives in `configs/corpus.yaml`.
2. Cleaning: normalize unicode (NFC), fix mojibake, collapse whitespace (keep paragraph
   breaks), drop non-English books' front matter, remove exact-duplicate paragraphs
   (hash-based). Log everything removed (counts).
3. **`data/clean/`** layout: `books/<slug>.txt`, `dictionary_prose.txt`, `dictionary.jsonl`,
   `manifest.json` (source URL, license, sha256, char/word counts per file).
4. Held-out split **by document**: reserve 1–2 whole books + 2% of dictionary entries as
   `data/clean/val/` (never seen in training — contamination lesson).
5. **`notebooks/01_corpus_stats.ipynb`**: char/word counts, estimated tokens (chars/4 rough +
   real count with GPT-2 tokenizer for calibration), composition pie, length histograms,
   most-common-words sanity check.

## Decision points

- Which books (user's taste — this is *their* model). Log the list.
- Dictionary rendering format for pretraining (prose template wording — show 2–3 templates,
  pick one; template diversity vs consistency is a real research question, note it for P5-G).
- Books:dictionary mixing ratio for the training stream (default: natural proportions; P5-G
  can ablate).
- Supplement yes/no now (can defer to phase 5/9).

## Learning checkpoints

- Why dedup matters (memorization, val contamination); why we split by document not by line.
- Chars-per-token intuition; why token counts differ across tokenizers.
- Why data quality/mixing beats architecture tweaks at small scale.

## Exit criteria
`scripts/build_corpus.py` rebuilds everything from scratch; manifest + stats notebook done;
PROGRESS/DECISIONS updated.
