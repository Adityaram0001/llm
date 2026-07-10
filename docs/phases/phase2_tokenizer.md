# Phase 2 — Tokenizers: from scratch AND off-the-shelf

**Goal:** implement BPE by hand (deep understanding), train a production-grade tokenizer on our
corpus with HF `tokenizers`, compare both against GPT-2's, choose one, and tokenize the corpus
to memmap files.
**Effort:** 1–2 sessions. This is a flagship learning phase — go slow, explain everything.

## Part A — BPE from scratch (learning artifact, notebook-first)

**`notebooks/02_bpe_from_scratch.ipynb`** + **`src/llmlab/tokenizer/bpe_scratch.py`**:

1. Implement byte-level BPE training: start from 256 byte tokens; count adjacent-pair
   frequencies; merge most frequent; repeat to target vocab. Pure Python; fine if it only
   handles a few MB — train it on ONE book.
2. Show intermediate state every N merges: what the first 20 merges are (spaces+common
   letters!), how "the", " and" emerge, when whole words appear.
3. Implement `encode()` (apply merges in rank order) and `decode()`. Round-trip test.
4. Pre-tokenization discussion & experiment: none vs whitespace-split vs GPT-2's regex —
   show how it changes learned merges (why you don't want merges crossing spaces).
5. Visualize: vocab-size vs corpus-compression curve (tokens needed to encode a held-out page
   as vocab grows 256→8k).

## Part B — Real tokenizer with HF `tokenizers`

**`src/llmlab/tokenizer/train_hf.py`**: byte-level BPE (GPT-2 style pipeline), trained on the
FULL corpus. Train three vocab sizes: **8k, 16k, 32k**. Special tokens: `<|endoftext|>` (+
reserve `<|user|>`, `<|assistant|>`, `<|pad|>` ids for phase 8 — cheap now, painful later).
Save to `data/tokenized/tokenizers/<name>/`.

## Part C — Comparison study (mini-experiment, registered in registry as p2 rows)

**`notebooks/03_tokenizer_compare.ipynb`**: for scratch-BPE(8k), HF 8k/16k/32k, GPT-2(50k):
- **Fertility** (tokens per word) on held-out books AND on dictionary text separately.
- Compression ratio (bytes/token); vocab utilization; rare-word handling (pick 20 obscure
  dictionary headwords — how do they split?); numbers & punctuation behavior.
- Trade-off discussion: bigger vocab = shorter sequences (faster, more context) but bigger
  embedding table (at 100M params, a 32k×768 embedding+unembedding ≈ 49M ≈ half the model!
  — show this math) and rarer-token undertraining.

## Decision point (big one — log as D-0xx)

Choose THE project tokenizer (default recommendation: **HF byte-level BPE, 16k vocab** —
balanced for a 100M model on a small corpus). Baseline configs reference it by path.

## Part D — Tokenize the corpus

**`scripts/tokenize_corpus.py`**: encode train/val corpus → `data/tokenized/<tok_name>/train.bin`,
`val.bin` as `uint16` numpy memmaps (+ `meta.json`: vocab size, doc boundaries, token counts).
Documents joined with `<|endoftext|>`. Verify by decoding random slices.

## Learning checkpoints

- Execute BPE merges on paper for a toy string; explain why byte-level never has OOV.
- Why does vocab size interact with model size? What's fertility and why does it matter for
  speed and for "how much text fits in context"?
- Why do we reserve chat special tokens before pretraining?

## Exit criteria
`train.bin`/`val.bin` exist for the chosen tokenizer; comparison notebook has figures + a
written verdict; decision logged; PROGRESS updated.
