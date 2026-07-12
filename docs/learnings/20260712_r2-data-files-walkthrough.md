# 2026-07-12 — What's actually in the R2 bucket: tokenizer artifacts vs tokenized data, and how mixing will work

## Context
After pushing `data/tokenized/` to the new R2 bucket (`llm`, D-026), the user asked for a
file-by-file walkthrough: what each of the 16 pushed files is, why it exists, and how it feeds
into pretraining — wanted a solid mental model before M/L-tier work starts, not just "it's there."

## The two groups of files

**Tokenizer artifacts** (`tokenizers/hf_bpe_{8k,16k,32k}/{vocab.json,merges.txt,tokenizer.json}`):
`vocab.json` is the token→ID map, `merges.txt` is the ordered BPE merge rules (the actual
compression algorithm), `tokenizer.json` is HF's serialized all-in-one form that code loads.
Only `hf_bpe_16k` is wired into any config — the 8k/32k folders are D-014's *comparison record*,
kept because the phase-2 tokenizer study measured all three head to head before picking:

| metric | 8k | 16k (chosen) | 32k |
|---|---|---|---|
| fertility, books (tok/word) | 1.612 | 1.500 | 1.427 |
| vocab utilization, held-out text | 87.7% | 71.3% | 49.3% |
| embed+unembed cost @ 100M budget (tied) | 6.1% | 12.3% | 24.6% |

The click-moment: every metric traces the same diminishing-returns curve. 8k→16k buys most of
the compression gain; 16k→32k buys much less while nearly doubling the embedding table's share
of the param budget, and over half of 32k's extra merges don't even fire on our own held-out
text — they're artifacts of the training corpus, not generalizable structure. 16k wins on
"most of the benefit, least of the cost," not on being universally correct.

**Tokenized data** (`hf_bpe_16k/*.bin`, `*.npy`, `meta.json`): all `.bin` files are flat
`uint16` numpy memmaps — one array of token IDs, 2 bytes each (16k vocab needs 2 bytes; uint8
maxes at 256). No object overhead — this is what lets `np.memmap` read a random slice without
loading the whole file into RAM.

| File | Tokens | Docs | Role |
|---|---|---|---|
| `train.bin` | 17,665,275 | 111 | Core corpus (books+dictionary) — the ONLY thing `p4_s_baseline` trained on |
| `val.bin` | 179,655 | 3 | Held-out eval split |
| `supplement_tinystories.bin` | 520,469,119 | 2,119,489 | Prepped, not yet used in any run |
| `supplement_fineweb.bin` | 992,803,683 | 808,365 | Prepped, not yet used in any run |
| `supplement_*_docstarts.npy` | — | — | Byte-offsets of each doc's start, for optional strict-boundary sampling |
| `meta.json` | — | — | vocab_size, eot_id, per-split token/doc counts + doc_starts |

## What clicked: `docstarts.npy` and the boundary-crossing default

`MixedSourceLoader` (`src/llmlab/data/loader.py`) defaults to letting a sampled window straddle
a document boundary — this is deliberate, the GPT-2 "concat-and-chunk" convention: the
`<|endoftext|>` token inside the window IS the model's only signal that two unrelated documents
just met, and that's real training signal (how to recognize a topic/register shift), not noise
to avoid. The alternative (`Source(respect_doc_boundaries=True, docstarts_path=...)`) exists
per-source for cases where straddling would hurt (e.g. short self-contained entries) — the
docstarts files are prepared for every supplement so that option is available later without
re-tokenizing, even though nothing currently turns it on.

## The Chinchilla arithmetic tying it together (why FineWeb-Edu exists at all)

L-tier active params ≈ 95.58M (D-015) → Chinchilla ~20 tok/param → needs **~2.1B tokens**.
Fresh pool: 17.7M (core) + 520.5M (TinyStories) + 992.8M (FineWeb) ≈ **1.53B tokens** — short of
2.1B on a single pass. Muennighoff et al. 2023's finding (repeating training data up to ~4
epochs doesn't meaningfully hurt) gives a ceiling of 1.53B × 4 ≈ **6.1B tokens** — a 2.9x margin
over the 2.1B need. Before FineWeb-Edu was added (D-015 originally), core+TinyStories alone gave
almost exactly zero margin at 4 epochs — that's the actual reason the FineWeb-Edu sample exists,
not just "more data is better."

## What's still open (not a gap, just not decided yet)

The per-source **mixing weights** (what fraction of each training batch comes from core vs.
TinyStories vs. FineWeb) are not chosen yet — S-tier's baseline used a single source
(`weight: 1.0`, no mixing at all). That decision slot is where RW-4's domain data
(finance/self-help — user still needs to pick PD titles) will also plug in, as a fourth weighted
`Source` in the same YAML list, no loader code changes needed. This is why the loader was built
generic (list of sources + weights) even though only one source has been exercised in a real run
so far.

## Related
[[../DECISIONS.md]] D-014 (tokenizer choice), D-015 (tier sizing + data budget), D-019/D-020
(supplement tokenization + FineWeb-Edu sizing), D-026 (R2 push). See also
`docs/learnings/20260711_gpu-vocab-datamix.md` for the earlier domain-mixing discussion this
connects to.
