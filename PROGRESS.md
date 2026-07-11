# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 4 — `docs/phases/phase4_training.md`
**Last session:** 2026-07-11 — Phase 3 completed: config-driven GPT in `src/llmlab/model/`
(`config.py`, `norms.py`, `positional.py`, `attention.py`, `ffn.py`, `block.py`, `gpt.py`).
Baseline path fully implemented (MHA/GQA/MQA via `F.scaled_dot_product_attention`, learned/
sinusoidal/rope/alibi/none positions, layernorm/rmsnorm, gelu/swiglu FFN, weight tying,
gpt2/scaled init); `attention="mla"`/`moe`/`mtp` are config fields that raise
`NotImplementedError` pointing to phase 5. Tier sizes finalized vocab-aware (**D-015**): S
9.71M / M 34.62M / L 104.80M (deep-narrow aspect, 24 layers × 576 for L), configs in
`configs/model_{s,m,l}.yaml`. Baseline hyperparameter defaults logged (**D-016**): tying ON,
head_dim=64 fixed, dropout=0.0, GPT-2 init. **Bug caught while building configs:** the phase-3
spec said vocab=16,384 (misreading "16k" as 2^14) — actual trained tokenizer is 16,000
(corrected in the spec, D-015, and the phase-3 parameter-allocation learning note).
`tests/test_model.py`: 51 tests green on both `mps` and `cpu` (shapes, causal-mask-doesn't-leak-
future-into-past for every pos_encoding, loss≈ln(vocab) at init, generate w/ top-k/top-p, every
config axis instantiates, tied-weights share storage, RoPE relative-shift property, deferred
techniques raise NotImplementedError). `notebooks/04_shapes_walkthrough.ipynb`: tensor-by-tensor
forward pass, param-budget pie charts across tiers, causal-mask visualization, untrained-model
attention heatmap, overfit-one-batch (loss 9.72→0.038 in 200 steps) — executes cleanly end to
end. Overfit-one-batch run was NOT registered in `experiments/registry.csv` (a debug sanity
check, not a comparable ablation — no baseline/hypothesis/val-loss); flagging this call in case
a future session disagrees.
**Open blockers:** none. RW-1 (tokenize supplement) now includes a FineWeb-Edu sample per
D-015 — needs its own >2GB download go-ahead when phase 4 gets there. The D-008 flag (hero run
≈ 1.5–3 weeks on the Mac) remains resolved in principle by **D-010**: rented RTX 5090 as burst
compute for M/L-tier runs (playbook `docs/CLOUD.md`, scripts in `scripts/cloud/`). Final
go/no-go + provider choice happens when the first big run is actually needed.

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | done |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | done |
| 2 | Tokenizers (scratch + HF) | `docs/phases/phase2_tokenizer.md` | done |
| 3 | Model architecture | `docs/phases/phase3_architecture.md` | done |
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

## Phase 3 checklist (done)

- [x] `src/llmlab/model/config.py`: `ModelConfig` dataclass (+ `MLAConfig`/`MoEConfig`/
  `MTPConfig`), `from_yaml`, validates `n_heads % n_kv_heads == 0` and MLA needs an `mla:` block
- [x] `norms.py` (LayerNorm/RMSNorm), `positional.py` (learned/sinusoidal/RoPE/ALiBi + relative-
  shift math), `attention.py` (MHA/GQA/MQA via SDPA, qk_norm, RoPE injection), `ffn.py`
  (GELU/SwiGLU), `block.py` (pre/post-norm residual wiring), `gpt.py` (embeddings→blocks→
  final norm→head; `forward`, `generate` w/ temperature+top-k+top-p, `num_params(breakdown=)`,
  `estimate_flops_per_token`)
- [x] `attention="mla"`, `moe`, `mtp` raise `NotImplementedError` (config fields exist, phase 5)
- [x] Tier sizes finalized vocab-aware, deep-narrow L-tier, FineWeb-Edu data-budget plan (D-015);
  baseline defaults tying/head_dim/dropout/init (D-016); `configs/model_{s,m,l}.yaml` committed
- [x] `tests/test_model.py`: 51 tests green on mps AND cpu
- [x] `notebooks/04_shapes_walkthrough.ipynb`: executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated (D-015, D-016); phase marked done

## Rework queue (see CLAUDE.md "Change management")

| ID | What | Why | Fix in phase | Status |
|----|------|-----|--------------|--------|
| RW-1 | Tokenize TinyStories supplement + a FineWeb-Edu sample (size/mixing ratio TBD) with hf_bpe_16k → `data/tokenized/hf_bpe_16k/supplement_*.bin`; extend `scripts/tokenize_corpus.py`. FineWeb-Edu download (>2GB expected) needs its own go-ahead per CLAUDE.md before pulling. **Per D-017: run entirely on the Mac (CPU-only, streaming/chunked — never hold corpus in RAM, append to memmap incrementally), then `scripts/cloud/data_push.sh` to the R2 bucket as the final step** | D-015: L-tier is 105M, needs ~2.1B tokens; repetition alone (~4 epochs of core+TinyStories) is right at the edge, so a FineWeb-Edu sample was chosen to add margin + topic diversity | 4 (before first M-tier run) | todo |
| RW-3 | One-time cloud accounts setup with the user: push repo to GitHub (private is fine), Docker Hub account, build+push `docker/Dockerfile` (buildx, linux/amd64), Cloudflare R2 bucket `llmlab` + rclone remote on Mac, provider pod template with env vars | D-017: Docker fast-start + bucket data logistics chosen for billed-time efficiency | 4 (any time before first cloud run; ~30 min, all free tiers) | todo |
| RW-4 | Domain corpus expansion (finance/self-help/wisdom): user picks PD-only books (Gutenberg-era finance/self-help classics — modern bestsellers are copyrighted), optionally + finance-filtered FineWeb-Edu slice; loader gets per-source mixing weights so domain share of the TRAINING STREAM (not disk) is explicit; keep domain repetition ≤~4 epochs. User's target: 10–20% (recommendation 15–25%); final % is the user's call when phase 4 builds the loader. Also: finance/wisdom probes in phase 6, domain-mix ablation in P5-G (specs updated) | User wants a finance/wisdom-steered model (2026-07-11 discussion, see `docs/learnings/20260711_gpu-vocab-datamix.md`) | 4 (loader + corpus) / 6 (probes) / 5-G (ablation) | todo |

## Parking lot (future ideas, deliberately not scheduled)

- **v2 scale-up** (after phase 9): 32k vocab + 160–180M params + ~3.2B tokens (1.6× data,
  correct Chinchilla coupling). Do NOT do mid-project: vocab change retokenizes everything and
  breaks ppl comparability with all v1 runs; 32k only pays once the corpus is big/diverse
  enough (phase 2 measured 49.3% vocab utilization at 32k on the v1 corpus). See
  `docs/learnings/20260711_gpu-vocab-datamix.md` §3.
| RW-2 | ~~Recompute D-008/D-010 if L-tier grows beyond ~105M~~ — resolved by D-015: L-tier stayed at ~105M (95.6M active), in-range of existing extrapolations, no recompute needed | D-015 finalized tier sizes vocab-aware | 3 | done |

## Run ledger (latest 10 — full list in experiments/registry.csv)

5 non-training comparison rows from the phase-2 tokenizer study (`20260710_p2_tokenizer-*`) —
see `experiments/registry.csv`. No model training has happened yet (phases 0-2 were
environment + data + tokenizer setup).

## Notes for next session

- Start with Phase 4. Read `docs/phases/phase4_training.md`. First real thing it needs is RW-1:
  tokenize the TinyStories supplement + decide/pull a FineWeb-Edu sample (D-015 chose this to
  close the ~2.1B-token gap for the L-tier hero run; sample size/mixing ratio wasn't decided,
  and the FineWeb-Edu download is >2GB so needs its own go-ahead per CLAUDE.md before pulling).
  Per D-017 tokenization is Mac-local CPU work (stream, chunk, memmap-append; ~under an hour);
  finish by pushing bins to the R2 bucket via `scripts/cloud/data_push.sh` (once RW-3 exists).
- RW-3 (one-time cloud accounts: GitHub remote, Docker Hub + image build, R2 bucket + rclone,
  pod template) is user-facing setup — walk it interactively when convenient, ideally before
  the first M-tier run. Cloud flow after that: `docs/CLOUD.md` "Docker fast-start" (billed
  cold start ≈ 2–4 min; big files move only via bucket, never rsync).
- Model is ready (phase 3, D-015/D-016): `src/llmlab/model/` (`GPT`, `ModelConfig`), configs at
  `configs/model_{s,m,l}.yaml` (S 9.71M / M 34.62M / L 104.80M, deep-narrow L-tier, vocab=16000,
  head_dim=64 fixed, tied embeddings, rmsnorm/pre-norm/rope/swiglu/gpt2-init defaults, dropout
  0.0). `tests/test_model.py` is the reference for how every config axis behaves — reuse the
  `tiny_config()` pattern for training-loop unit tests rather than re-deriving fixtures.
  `notebooks/04_shapes_walkthrough.ipynb` has the tensor-shape reference if a training bug needs
  shape-by-shape debugging. Remember: `attention="mla"`, `moe`, `mtp` configs raise
  `NotImplementedError` — don't reach for them before phase 5.
- Tokenizer is decided (D-014): **HF BPE, 16,000 vocab** (corrected from an earlier "16,384"
  typo carried in the phase-3 spec — see D-015's correction note; the real tokenizer/data always
  used 16,000). Files at
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
