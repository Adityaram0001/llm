# DECISIONS — project decision log

> Append-only. Every non-obvious choice gets an entry — that's a core requirement of this
> project. A decision is "non-obvious" if a future reader could reasonably ask "why?".
>
> Format:
> ```
> ## D-NNN — <title>  (YYYY-MM-DD, phase N)
> **Decision:** what was chosen.
> **Options considered:** A vs B vs C, one line each.
> **Why:** the reasoning, incl. hardware/learning-value trade-offs.
> **Revisit if:** condition under which this should be reconsidered.
> ```

## D-001 — Model scale strategy: 3 tiers, not one 100M model  (2026-07-10, planning)
**Decision:** Three model tiers sharing one codebase/config system: **S ≈ 10M** (ablation
workhorse, minutes–1h runs), **M ≈ 30–50M** (confirmation runs, few hours), **L ≈ 100–125M**
(single "hero" pretrain at the end, 1–3 days).
**Options considered:** (a) only 100M — every experiment takes a day+, kills the fast-feedback
learning loop; (b) only tiny models — never experience real-scale pain; (c) tiered — do both.
**Why:** On an M4 GPU a 100M model trains at roughly a few thousand tokens/sec; most published
ablations (nanoGPT speedruns, scaling-law papers) run the same way: sweep small, confirm big.
**Revisit if:** Phase 0 benchmark shows throughput very different from assumptions.

## D-002 — Architecture family: decoder-only GPT, config-driven variants  (2026-07-10, planning)
**Decision:** Single decoder-only transformer implementation in `src/llmlab/model/` where every
studied technique (norm type, activation, positional encoding, attention variant, MoE, MTP …)
is a config flag — not separate model files.
**Why:** The whole point is A/B-ing techniques; config flags make ablations one-line diffs and
keep the experiment registry meaningful.
**Revisit if:** a technique (e.g. Mamba) genuinely can't share the skeleton — then a sibling module.

## D-003 — Dictionary source: NOT Oxford  (2026-07-10, planning)
**Decision:** Use public-domain/free dictionary data: primary = **Webster's 1913 / GCIDE**
(classic full English dictionary, public domain); optional supplement = Wiktionary extract
(kaikki.org JSON) and WordNet glosses.
**Options considered:** Oxford (user's first idea) — copyrighted, scraping violates ToS and it
isn't downloadable; Webster's 1913 — free, similar coverage/register; Wiktionary — free, modern,
JSON-structured, bigger but noisier.
**Why:** Same learning value (definitional knowledge in pretraining + "define X" eval probes)
with zero legal problems.
**Revisit if:** user obtains licensed dictionary data some other way.

## D-004 — Data factory: human-in-the-loop DeepSeek web workflow, no browser automation  (2026-07-10, planning)
**Decision:** `tools/data_factory/` implements: prompt-batch generator → user manually pastes
batches into DeepSeek web chat and saves replies to `inbox/` → parser validates/repairs JSON →
retry queue for failures. Optional backend: official DeepSeek API (very cheap, ~$0.3–0.6 per
million tokens) behind the same interface. Local Ollama model as third backend.
**Options considered:** (a) Selenium/Playwright bot on the web UI — violates DeepSeek ToS,
brittle, account-ban risk; (b) manual-paste batch pipeline — free, ToS-clean, ~10 min of human
time per few hundred Q&A pairs; (c) paid API — costs pennies at our scale.
**Why:** (b) gives the free workflow the user wanted without ToS violation; (c) exists as a
flag-switch when volume grows.
**Revisit if:** DeepSeek publishes an official free programmatic tier.

## D-005 — Tracking stack: wandb + local JSONL, CSV registry  (2026-07-10, planning)
**Decision:** Weights & Biases (free tier, project `llm-lab`) for live curves/system metrics;
every run ALSO writes `metrics.jsonl` + `config.yaml` locally; `experiments/registry.csv` is the
comparison index. TensorBoard not used.
**Why:** wandb = best live-run UX + teaches an industry tool; local files = offline-safe source
of truth; one CSV = trivially pandas-comparable across dozens of runs.
**Revisit if:** wandb free tier limits bite — fall back to trackio or pure-local + notebook dashboards.

## D-006 — Corpus: Gutenberg books + dictionary + TinyStories supplement  (2026-07-10, planning)
**Decision:** Core corpus = user-picked Project Gutenberg books (~5–15M tokens) + dictionary
(~10–20M tokens). Because 100M params wants ~2B tokens (Chinchilla) and we'll have ~30M, add an
optional supplement for the bigger runs: **TinyStories** (~500M tokens, synthetic, clean) and/or
a small **FineWeb-Edu** sample. Small-tier ablations can run books-only.
**Why:** Books+dictionary alone means heavy multi-epoch training → an overfitting lab (itself a
lesson, phase 5 studies it explicitly), but the hero run needs more tokens to be interesting.
**Revisit if:** user prefers a strictly books+dictionary purist model — that's a valid capstone choice.

## D-007 — Python/venv/layout  (2026-07-10, planning)
**Decision:** Python 3.11+, `python -m venv .venv`, `pip`, editable install of `src/llmlab`
via `pyproject.toml`. Scripts in `scripts/` for anything long-running; notebooks for exploration.
**Why:** venv was a user requirement; src-layout keeps notebook imports honest.

## D-008 — Compute budget: measured MPS throughput (2026-07-10, phase 0)
**Decision:** Use these measured numbers (M4, `torch==2.13.0`) to calibrate every later time
estimate. Bench: `scripts/bench_mps.py`, TinyGPT ~9.1M params (6 layer, d_model=256, n_head=4,
manual QKV + `F.scaled_dot_product_attention`, bf16 autocast, weight-tied embeddings).

**Matmul TFLOPS (square, bf16):** ~3.5–3.8 TFLOPS across 1k–4k. fp32 is ~1.9–3.3 TFLOPS — bf16
autocast gives a real, if modest, speedup on this GPU; nowhere near quoted M4 peak FLOPS because
this is unfused eager-mode matmul, not a tuned kernel.

**TinyGPT (~9.1M params) fwd+bwd tokens/sec, best-per-seq_len (sweet-spot micro-batch):**
| seq_len | micro-batch | tokens/sec | mps_alloc |
|---------|-------------|------------|-----------|
| 256     | 16          | ~23,800    | 160MB     |
| 512     | 8           | ~20,800    | 163MB     |
| 1024    | 16          | ~15,500    | 548MB     |

CPU comparison (bs=4, seq=256): MPS ~22,500 tok/s vs CPU ~15,700 tok/s — MPS wins but by a much
smaller margin than expected for a discrete GPU; at this tiny scale kernel-launch overhead and
unified-memory traffic dominate over raw compute.

**Important finding — throughput cliff well below any advertised memory ceiling:** sweeping
micro-batch size does NOT degrade gracefully into a clean OOM. Tokens/sec is flat (~20-24k)
across most of the batch sweep, then falls off a cliff (3-15x slower) at a specific
`mps_alloc` size *before* a hard `RuntimeError` OOM ever fires — e.g. seq=512 dropped from
20,802→1,922 tok/s going bs=32→64 (mps_alloc 535MB→1037MB); seq=1024 dropped from 15,485→1,435
tok/s going bs=16→32 (mps_alloc 548MB→1037MB). This ceiling (~1GB `mps_alloc`) is nowhere close
to either total system RAM (16GB) or `torch.mps.recommended_max_memory()` (12.7GB reported) —
it looks like an MPS allocator/Metal-heap fragmentation effect, not a real capacity limit.
**Practical rule:** tune micro-batch to the plateau, not to the edge of what fits — there is no
warning between "fine" and "10x slower," and it arrives well before an actual OOM crash.

**Estimated wall-clock (fwd+bwd only, no optimizer/data/logging overhead — treat as an optimistic
floor, real scripts will be slower):**
- 10M-tier model × 100M tokens, seq_len 512 @ ~20.8k tok/s → ~4,800s ≈ **1.3 hours**.
- 100M-tier model × 1B tokens: FLOPs scale ~linearly with params (Chinchilla-style 6ND), so
  extrapolate tok/sec down by ~11x (100M/9.1M) → ~1,900 tok/s → 1e9/1900 ≈ 528,000s ≈ **~6.1
  days**, before adding optimizer-step, data-loading, and checkpoint overhead.
**Why this matters:** D-001 assumed the 100M "hero run" takes 1-3 days; D-006 assumed ~2B
Chinchilla-optimal tokens for the 100M model. Those two assumptions are in tension with this
measurement — 1B tokens alone extrapolates to ~6 days of pure compute, so 2B tokens end-to-end
is more likely **1.5-3 weeks**, not "1-3 days." Flagged to the user; not resolved here.
**Revisit if:** phase 4's first real training script measures actual (optimizer-included)
tokens/sec at the true model size — replace this extrapolation with a direct measurement, and
revisit D-001/D-006's token budget or timeline expectations if the gap holds.

## D-009 — Torch version and wandb default mode (2026-07-10, phase 0)
**Decision:** `torch==2.13.0` (latest stable satisfying the pinned `requirements.txt` range
`>=2.7,<3`), installed via `scripts/setup.sh` — includes recent MPS backend fixes over 2.7 baseline.
wandb defaults to **offline mode** (`WANDB_MODE=offline` set in training scripts); `wandb sync`
can push a run later if the user wants the hosted dashboard.
**Options considered:** wandb online-by-default — needs `wandb login` + connectivity on every
run; offline-by-default — zero-setup, matches D-005 ("wandb can be offline; local files are
the source of truth"), user can sync selectively.
**Why:** User confirmed offline-by-default. Removes a login dependency from every training
script and keeps local `metrics.jsonl` as the always-available record.
**Revisit if:** user wants live remote monitoring during long runs (e.g. the multi-day 100M
hero run in D-008) — flip the default or sync proactively for that specific run.

## D-010 — Hybrid compute: Mac primary, rented RTX 5090 burst option  (2026-07-10, planning)
**Decision:** Keep the M4 Mac as the primary lab (all dev, notebooks, S-tier ablations). Add
rented cloud GPUs (RunPod recommended; gpuhub also fine; RTX 5090 class) as an *optional* burst
target for M/L-tier confirmations and the phase-9 hero run. Full playbook: `docs/CLOUD.md`;
helper scripts: `scripts/cloud/` (sync_up / remote_setup / sync_down). All project code must be
device-agnostic per CLOUD.md's portability rules (`get_device()` = cuda > mps > cpu,
`autocast_ctx()`, guarded backend calls, config-keyed DataLoader settings).
**Options considered:** (a) Mac-only — D-008 showed the 100M×2B-token hero run ≈ 1.5–3 weeks:
technically possible, practically miserable; (b) cloud-only — loses the always-free local loop
and the MPS learning angle; (c) hybrid — free fast iteration locally, ~$10–20 overnight hero run.
**Why:** A 5090 (~32GB VRAM, ~100+ bf16 TFLOPS vs the M4's measured ~3.6) is roughly 30–60×
effective throughput once torch.compile/bigger batches are on; the D-008 tension (hero-run
timeline vs token budget) dissolves for the cost of a dinner. Ablation wall-clock comparisons
must stay same-hardware; tokens-based curves remain comparable across machines.
**Revisit if:** first real cloud run measures very different $/token than estimated, or
spot-instance interruptions prove painful (then: on-demand only, or chunked WSD-style runs).

## D-011 — Book corpus: 112-book philosophy/classics selection via Gutenberg catalog  (2026-07-10, phase 1)
**Decision:** Final book list = 112 Project Gutenberg texts (~14.9M est. tokens post-cleaning),
listed in `configs/corpus.yaml`. User specified 20 seed authors (Marcus Aurelius, Seneca,
Epictetus, Lao Tzu, Confucius, Plato, Aristotle, Emerson, Thoreau, Sun Tzu, Descartes, Locke,
Spinoza, Schopenhauer, William James, Bacon, Montaigne, Voltaire, Rousseau, Mill), then
delegated the rest ("add 100 more books you decide, same genre; if still under 15M tokens add
50 more"). Rather than hand-picking 100+ titles from memory (real hallucination risk on
Gutenberg IDs at that scale), downloaded Gutenberg's full catalog metadata
(`pg_catalog.csv`, ~90k rows) as ground truth: filtered to English-language works in the
"Category: Philosophy & Ethics" bookshelf (2,519 candidates), then curated ~90 additional
works from ~23 canonical philosophers not yet covered (Kant, Hume, Hegel, Nietzsche, Hobbes,
Berkeley, Adam Smith, Burke, Paine, Pascal, Cicero, Plutarch, Boethius, Augustine, Machiavelli,
Erasmus, Xenophon, Diogenes Laertius, Zhuangzi, Dewey, Spencer, Santayana, Russell) plus
second/third works by several seed authors (more Plato dialogues, Aristotle's Politics/Poetics,
etc.). Verified real file sizes via HTTP HEAD on all 112 URLs before finalizing — estimated
~15.6M tokens (bytes/4), already over the 15M target, so the "+50 more" contingency wasn't
triggered.

Per-author calls flagged by research and resolved by user: **Seneca** — Aubrey Stewart's direct
translation of the Minor Dialogues (GB 64576), not L'Estrange's 17th-c. condensed paraphrase.
**Descartes** — both Discourse on Method (GB 59) and the separately-published Six Metaphysical
Meditations (GB 70091); no single GB text bundles them. **Locke** — both volumes of An Essay
Concerning Human Understanding (GB 10615 + 10616), not the shorter Second Treatise.

Held out as whole-document val split (never seen in training): Boethius' *Consolation of
Philosophy* (GB 14328) and Epictetus' *Enchiridion* (GB 45109) — both short, self-contained,
not part of any multi-volume split.
**Options considered:** (a) hand-pick from memory — rejected, high risk of wrong/hallucinated
Gutenberg IDs at this scale; (b) programmatic catalog-based selection — chosen, every ID
verified to exist and match its title before download.
**Why:** Matches phase-1's requirement for a repeatable, verifiable build; catalog-driven
selection scales to "however many books" without a manual-research bottleneck while still
respecting the user's stated genre.
**Revisit if:** user wants to prune the auto-selected ~90 further, or wants other genres added.

## D-012 — Dictionary source: GNU GCIDE tarball, not raw Gutenberg Webster's text  (2026-07-10, phase 1)
**Decision:** Downloaded the canonical GNU GCIDE distribution (`gcide-0.53.tar.xz` from
ftp.gnu.org, ~14MB, 23 letter files) rather than Project Gutenberg's own "Webster's Unabridged
Dictionary" text (GB 247/660-673, OCR'd free-form text). Parsed GCIDE's SGML-tagged entries
(`<p><ent>WORD</ent>...<hw>...</hw> <pos>...</pos> <def>...</def></p>`) with targeted regex —
not a real XML parser, since GCIDE's tags are unclosed self-closing style (`<br/`, `<ae/`),
which is invalid XML. 119,984 entries parsed. Bold-term prose template per user's choice
(`**word** (pos): def1; def2.`). 2% of entries (3,259) held out as val
(`random.Random(42)` shuffle, restored to alphabetical order within each split).
**Options considered:** (a) raw Gutenberg Webster's text — free-form OCR, no structured
word/pos/definition boundaries, fragile heuristic parsing; (b) GNU GCIDE tarball — same
underlying Webster's 1913 base text plus WordNet 1.5 supplement, already tagged per-entry.
Chose (b).
**Why:** GCIDE is what D-003 named as the target source; the tagged format parses far more
reliably than OCR'd text, and yields a clean structured `dictionary.jsonl` for eval-probe use
(phase 6/7) essentially for free.
**Bug caught during build:** GCIDE's `<hw>` tag carries pronunciation/stress markup (backticks,
quote marks, e.g. `` Ab"sinth` ``) meant for syllabification, not the clean headword — an
initial version preferred `<hw>` and leaked stress marks plus unclosed entity tags (`<ae/>` for
æ, etc.) into headwords. Fixed to prefer `<ent>` (the clean form), `<hw>` only as fallback, plus
a sweep for leftover unclosed self-closing tags.
**Revisit if:** WordNet-sourced entries (modern vocabulary mixed into GCIDE alongside 1913
Webster) prove undesirable for a "period" feel — could filter by the `<source>` tag to
1913-Webster-only entries.

## D-013 — TinyStories supplement downloaded in phase 1, not deferred  (2026-07-10, phase 1)
**Decision:** Downloaded `roneneldan/TinyStories` (HF `datasets`, streamed row-by-row to
`data/clean/supplement/tinystories.txt`, never accumulated in a Python list) now rather than
deferring to phase 4/5/9. Actual size: 2.12M stories, ~372M words, ~1.8GB on disk (~475-533M
tokens depending on estimator) — under CLAUDE.md's 2GB ask-first threshold but closer to it
than the ballpark given when the user approved ("~500MB-1GB" at the time of asking).
**Why:** User explicitly chose "download now"; actual size came in larger than estimated but
still under the 2GB threshold, so no separate re-confirmation was sought.
**Revisit if:** disk pressure becomes an issue (unlikely against the 512GB budget) or the
supplement mixing ratio needs tuning — it's stored separately today, not mixed into the
books:dictionary training stream (see `configs/corpus.yaml` `supplement.tinystories`).

## D-014 — Project tokenizer: HF byte-level BPE, 16k vocab  (2026-07-10, phase 2)
**Decision:** The project tokenizer is **HF byte-level BPE at 16k vocab**
(`data/tokenized/tokenizers/hf_bpe_16k/`), trained on the S-tier corpus (books train split +
`dictionary_prose.txt`, per D-006) via `src/llmlab/tokenizer/train_hf.py`
(`ByteLevelBPETokenizer`, `add_prefix_space=False`, `min_frequency=2`). Special tokens:
`<|endoftext|>`, plus `<|pad|>`/`<|user|>`/`<|assistant|>` reserved now for phase 8 chat
fine-tuning so their IDs never shift later. Corpus tokenized to
`data/tokenized/hf_bpe_16k/{train,val}.bin` (uint16 memmap) via `scripts/tokenize_corpus.py`.

**Options considered (measured in `notebooks/03_tokenizer_compare.ipynb`, held out of ALL
training):** HF BPE at 8k / 16k / 32k, GPT-2's own 50k-vocab tokenizer (reference, never
trained on our data), and the from-scratch BPE from notebook 02 (8k vocab, but trained on
only one book — pure-Python training can't afford the full corpus, so not a fair candidate,
included for completeness only).

| metric | 8k | 16k | 32k | gpt2 (50k) |
|---|---|---|---|---|
| fertility, held-out books (tok/word) | 1.612 | 1.500 | 1.427 | 1.469 |
| fertility, held-out dictionary | 2.068 | 1.927 | 1.815 | 1.829 |
| vocab utilization on held-out text | 87.7% | 71.3% | **49.3%** | 30.6% |
| avg pieces per obscure headword (n=20) | 4.40 | 3.80 | 3.45 | 3.45 |
| embed+unembed cost, tied/untied, @d_model=768, 100M budget | 6.1/12.3% | 12.3/24.6% | **24.6/49.2%** | 38.6/77.2% |

**Why:** Every metric shows the same diminishing-returns shape: 8k->16k buys most of the
fertility and rare-word gain (e.g. fertility +7.0% on books, rare-word pieces -0.60), and
16k->32k buys much less on top (+4.9%, -0.35) while roughly doubling the embedding-table's
share of a 100M-param budget (12-25% at 16k vs 25-49% at 32k, tied/untied) and dropping vocab
utilization on our own held-out text to under half (32k's extra merges are largely
document-specific artifacts that don't generalize even within our corpus). GPT-2's 50k vocab
confirms the domain-mismatch risk of an off-the-shelf tokenizer: only 30.6% of it ever fires
on our philosophy/dictionary text. User reviewed this table and chose 16k over 8k/32k.
**Revisit if:** phase 3's actual `d_model`/param budget ends up far from the 768/100M used
for the illustrative embedding-cost math, or a later phase adds a very different-domain data
source (code, modern web text) where 16k's fertility on that domain turns out poor.

## D-015 — Tier sizes finalized vocab-aware; L-tier is deep-narrow ~105M; data budget closed via FineWeb-Edu sample  (2026-07-11, phase 3)
**Decision:** Supersedes D-001's provisional tier sizes (written pre-tokenizer). All three
tiers fixed at `vocab=16,000` (D-014's actual trained tokenizer size — see the correction note
below), tied embeddings, `head_dim=64` throughout:

| Tier | d_model | layers | heads | embed | active | total (tied) | embed % |
|------|---------|--------|-------|-------|--------|--------------|---------|
| S | 192 | 15 | 3 | 3.07M | 6.64M | 9.71M | 31.6% |
| M | 320 | 24 | 5 | 5.12M | 29.50M | 34.62M | 14.8% |
| L | 576 | 24 | 9 | 9.22M | 95.58M | 104.80M | 8.8% |

**Bug caught while writing configs:** the phase-3 spec text (and this entry's first draft) said
"vocab is now fixed at 16,384" — treating "16k" as the power-of-two 2^14. The tokenizer
actually trained in phase 2 (`src/llmlab/tokenizer/train_hf.py`'s `--vocab-sizes` default,
confirmed in `data/tokenized/hf_bpe_16k/meta.json` and `tokenizers/hf_bpe_16k/vocab.json`) used
the literal round number **16,000**. Caught by cross-checking the real tokenizer file before
finalizing configs rather than trusting the spec's number. `docs/phases/phase3_architecture.md`
corrected in place (factual error, not a re-litigated decision). All tables/configs below use
the correct 16,000.

L-tier aspect ratio is **deep-narrow** (24 layers × 576, MobileLLM/SmolLM2-style) over
wide-shallow (GPT-2-style, e.g. 11×832 for the same 105M) — user's call after reviewing both
families at 105M/125M/160M candidates (all computed with the same head_dim=64, tied-embedding
formula). S and M kept the same depth-leaning philosophy for family consistency (one codebase,
consistent design language across tiers) rather than switching aspect families by tier.

**Data budget:** Chinchilla ~20 tok/param → L-tier needs ~2.1B tokens. Available fresh data
(17.7M core + ~500-533M raw TinyStories, untokenized) tops out at ~517-551M raw tokens — even
at Muennighoff's supported ~4 repeated epochs that's only ~2.1B, right at the edge with zero
margin. User chose to **add a FineWeb-Edu sample** (general web/edu text, HF
`HuggingFaceFW/fineweb-edu`) rather than lean entirely on repetition, to keep epoch count
comfortably low and add topic diversity beyond philosophy+children's-stories. Concrete sample
size/mixing ratio is a **phase-4 decision** (that's where the DataLoader and RW-1's
tokenization work happen) — this entry only fixes the *strategy*; the actual download (>2GB
expected) needs its own go-ahead per CLAUDE.md's data-budget rule before it happens.

**Options considered:** L-tier 105M / 125M / 160M (need ~2.1B/2.5B/3.2B tokens respectively —
125M and especially 160M pushed epoch count past the well-tested ~4x range without new data);
wide-shallow vs deep-narrow aspect at fixed budget; data-gap-closing via (a) repetition only,
(b) FineWeb-Edu addition — chosen, (c) deliberate undertraining at a larger size.
**Why:** 105M keeps the data story clean (comfortable epoch count even before the FineWeb-Edu
top-up); deep-narrow matches current small-model literature and gives the user a genuinely
different shape than GPT-2 to learn from, with wide-shallow preserved as the P5-G ablation
comparison at cheap S-tier cost instead of being the one-shot hero run's bet.
**Impacts:** RW-2 (recompute time/cost — done here, folded into this entry) resolved: 105M is
in-range of D-008/D-010 extrapolations, no timeline blowup. RW-1 updated: tokenization must now
include a FineWeb-Edu sample, not just TinyStories, sized/mixed in phase 4.
**Revisit if:** phase 4's actual DataLoader mixing design finds the FineWeb-Edu sample changes
these numbers meaningfully, or eval (phase 6) shows the deep-narrow shape underperforming badly
enough to swap to wide-shallow for a re-run.

## D-016 — Baseline architecture defaults: weight tying ON, head_dim=64 fixed, dropout=0.0, GPT-2 init  (2026-07-11, phase 3)
**Decision:** Applied the phase-3 spec's own recommendations as the baseline `configs/model_{s,m,l}.yaml`
defaults (also `src/llmlab/model/config.py`'s dataclass defaults, so a bare `ModelConfig(...)`
matches):
- **`tie_embeddings: true`.** Press & Wolf '16: sharing the input embedding and output
  unembedding matrix is a straightforward win — it's the same "which vector represents token
  X" mapping either direction, halves the embedding budget, and per
  `docs/learnings/20260711_parameter-allocation.md` matters MORE at our 16k vocab / small-model
  scale than at GPT-2/GPT-3 scale (untied would cost the L-tier model an extra 9.4M params —
  ~9% of the total budget — for zero known benefit).
- **`head_dim: 64` fixed** (not derived as `d_model // n_heads` with a free head count). 64 is
  the value nearly every reference model converges on (GPT-2, LLaMA, Mistral); fixing it makes
  `n_heads` the derived quantity instead, which keeps attention FLOPs/head comparable across
  tiers and avoids accidentally shrinking heads into an uninformative dimension as models get
  wider at a fixed head count.
- **`dropout: 0.0`.** Standard for modern LLM pretraining at these token budgets (GPT-3, LLaMA,
  Chinchilla-era models all train without dropout) — dropout was designed for the small-data,
  many-epoch regime; at internet-scale (or our repeated-epoch-but-large) token counts the model
  doesn't see the same example enough times for dropout's regularization to be the right tool,
  and it costs some final loss. Revisit only if the FineWeb-Edu-extended corpus still ends up
  heavily repeated (many >4 epochs) and overfitting shows up in val loss.
- **`init: gpt2`** (Radford et al. '19): all weights ~ N(0, 0.02), then every sub-layer's
  *residual-writing* projection (`attn.o_proj`, `ffn`'s final down-projection) additionally
  scaled by `1/sqrt(2*n_layers)`. The 0.02 constant is empirical (small enough that logits stay
  near-uniform at init regardless of d_model, since it isn't fan-in-scaled — verified in
  `tests/test_model.py::test_loss_near_ln_vocab_at_init`, loss lands within 0.01 of ln(vocab)).
  The 1/sqrt(2n) scaling exists because each block adds two roughly-independent contributions
  (attention out, FFN out) to the residual stream; without down-scaling them, the residual
  stream's variance grows ~linearly with depth, which destabilizes deep transformers at
  initialization. An alternate `init="scaled"` (uniformly applies 1/sqrt(2n) to every linear
  layer, not just residual-writing ones) is implemented too — both are real, testable code
  paths (`tests/test_model.py::test_init_axis_instantiates`), not just config placeholders, so
  init scheme is available as a P5 ablation without extra work later.
**Options considered:** untied embeddings (rejected, D-014/parameter-allocation math above);
derived head_dim (rejected, less comparable across tiers); dropout 0.1 (rejected, matches
neither modern practice nor our token-budget regime); "scaled" init as the default instead of
"gpt2" (rejected — GPT-2 init is the better-established default to build confidence in the
training loop against; "scaled" kept available for comparison).
**Why:** These are exactly the phase-3 spec's own recommendations; applied directly as the
logged default per CLAUDE.md's teaching-mode clause ("apply the logged default") since spec
already gave the trade-off reasoning — the round-trip questions this session were reserved for
the higher-uncertainty tier-size/aspect-ratio/data-budget choices (D-015).
**Revisit if:** phase 4 training shows instability at init that GPT-2 init should have
prevented (check the "scaled" variant), or val loss shows the corpus's repetition count made
dropout=0.0 a mistake.

## D-017 — Cloud packaging & data logistics: Docker image + R2 bucket; tokenize locally  (2026-07-11, pre-phase-4)
**Decision:** (a) Custom Docker image (`docker/Dockerfile`, deps-only, CUDA-12.8 torch base for
the sm_120 RTX 5090) built from the Mac with `buildx --platform linux/amd64`, pushed to Docker
Hub; pods start from a provider template with env vars, and `docker/entrypoint.sh` clones the
repo + pulls data automatically → billed cold-start ≈ 2–4 min. Rebuild only on dependency
changes; code moves via git, data via bucket. (b) Tokenized `.bin` shards live in a
**Cloudflare R2** bucket (free 10GB tier, zero egress fees), pushed once from the Mac via
`scripts/cloud/data_push.sh` (rclone), pulled by pods at datacenter speed. Checkpoints during
multi-day runs also rclone'd to the bucket from the pod. (c) **All tokenization happens on the
Mac** — BPE encoding is multithreaded CPU (Rust) work, ~under an hour for the full ~2.1–2.5B
tokens, GPU irrelevant; pods only ever download finished bins.
**Options considered:** data baked into the image (bloated, rebuild per data change, slow
registry pulls) vs rsync from Mac each rental (5GB over home upload = 20–45 min of billed pod
time) vs object storage (upload once off-clock, <1 min pulls, free egress on R2) — bucket wins;
stock pod image + pip-on-boot (5–15 min billed, drifting versions) vs custom image — image wins
for repeated rentals, rsync flow kept as fallback; B2/S3 vs R2 — R2's zero egress suits
repeated pulls.
**Impacts:** CLOUD.md gained three sections (Docker fast-start, Data logistics, Tokenization
is CPU work); repo needs a GitHub remote + Docker Hub account (one-time user setup, phase-4
session walks through it); RW-1 now specifies local tokenization + `data_push.sh` as its final
step.
**Revisit if:** R2 free tier is exceeded (hero-scale data + checkpoints may brush 10GB — prune
old checkpoint copies or pay ~$0.015/GB/mo) or a provider's network-volume ends up cheaper for
a long experiment series pinned to one region.

## D-018 — GPU rental sizing + batch-size policy: rent $/FLOP not VRAM; calibrate once, never auto-adjust  (2026-07-11, pre-phase-4)
**Decision:** (a) Default rental = RTX 5090; RTX PRO 6000 (96GB) rejected for now — same
Blackwell/sm_120 generation (same image, same code, zero changes to switch), but our 105M
model + optimizer + activations uses a few GB, so 96GB VRAM at ~2× the hourly rate buys
nothing. Re-evaluate only if a workload is actually VRAM-bound. (b) Batch-size policy:
`scripts/find_batch_size.py` (phase-4 deliverable) calibrates micro-batch per hardware in a
~2-min pre-run sweep; **effective batch is fixed in config, micro_batch × grad_accum
re-factorizes it per machine; no dynamic/runtime batch adjustment** — effective batch is a
hyperparameter, drifting it mid-run destroys ablation comparability. Monitoring = tokens/sec
in metrics.jsonl (ground truth) + wandb system charts (GPU util/power/VRAM) when online.
**Why:** utilization% can read high while bandwidth-bound — tokens/sec is what we pay for;
phase 0's MPS throughput-cliff finding showed measuring beats assuming on every new hardware.
**Revisit if:** models grow past ~1B-param scale (VRAM starts mattering) or variable-length
SFT batching (phase 8) wants token-count-based bucketing — that's a different, legitimate
kind of dynamic batching.

## D-019 — Bug fix: TinyStories supplement had ambiguous story boundaries; fixed and retokenized  (2026-07-11, phase 4)
**Decision:** Fixed `acquire.build_tinystories_supplement` (phase 1) and regenerated
`data/clean/supplement/tinystories.txt` + `data/tokenized/hf_bpe_16k/supplement_tinystories.bin`.

**Bug caught while building RW-1's streaming tokenizer:** the phase-1 writer joined stories with
`story + "\n\n"`, treating a blank line as "end of story." But 94% of TinyStories rows contain
their own internal blank-line paragraph breaks (measured on a 2,000-row sample: 1,881/2,000),
making inter-story and intra-story blank lines indistinguishable in the flat file. A first
streaming-tokenizer pass that split on every blank line produced 11,254,913 "documents" against
the ~2.12M real stories logged in D-013/`manifest.json` — a >5x inflation — and, worse, inserted
spurious `<|endoftext|>` tokens mid-story, teaching a false "story ends here" signal throughout
the supplement. Caught by cross-checking the tokenizer's `n_docs` output against D-013's known
story count rather than trusting the first run. A second, unrelated latent bug in the same
function was also fixed in passing: `build_tinystories_supplement(force=False)` on an existing
file returned zero-valued stats (`text.read_text()[:0]`) instead of real ones — harmless so far
since the original manifest entry was written on a force=True run, but would have silently
zeroed the manifest on any future non-force re-run.

**Fix:** `acquire.py` now collapses internal blank-line breaks to single newlines
(`re.sub(r"\n\s*\n+", "\n", story)`) before writing, so `"\n\n"` means *only* "story boundary"
in the output file — the format any blank-line-delimited streaming reader depends on. Also fixed
the stats-on-skip bug to read the real file. Regenerated from the HF-cached dataset (no
re-download, ~20s) via `scripts/build_corpus.py --skip-books --skip-dictionary --force`:
`n_stories` unchanged (2,119,489, matches D-013), `chars` dropped slightly (1,899,973,203 →
1,890,823,551 — the collapsed whitespace), `words` unchanged. Retokenized:
`data/tokenized/hf_bpe_16k/supplement_tinystories.bin` now has 2,119,489 docs / 520,469,119
tokens (up from the discarded first pass's 518,765,711 tokens under the wrong 11.25M-doc split —
extra `<|endoftext|>` tokens previously ate into the story text's token budget). Verified: decoded
doc 0 now spans the Lily/needle story's full 3 paragraphs ending in one real EOT, not truncated
at the first internal blank line.
**Impacts:** none outside phase 4/RW-1 — no training has consumed the old (wrong) supplement bin
yet.
**Revisit if:** a future supplement source (e.g. FineWeb-Edu) uses a similar flat blank-line-
delimited format — apply the same "confirm document boundaries independently, don't trust a
delimiter that could also occur inside a document" lesson before writing its streaming tokenizer.

## D-020 — FineWeb-Edu sample: ~1B tokens (sample-10BT config, 3.6GB text), RW-1 tokenization complete  (2026-07-11, phase 4)
**Decision:** Downloaded and tokenized RW-1's FineWeb-Edu top-up. `acquire.build_fineweb_edu_supplement`
streams `HuggingFaceFW/fineweb-edu` (`sample-10BT` config, `streaming=True` so only the shards
needed are fetched) into `data/clean/supplement/fineweb_edu.txt`, stopping at a
`target_bytes` cap (`configs/corpus.yaml` `supplement.fineweb_edu.target_bytes`). Applies the
same D-019 fix proactively: each row's internal blank-line paragraph breaks are collapsed to
single newlines before writing, so `"\n\n"` means only "document boundary." Tokenized via
`scripts/tokenize_corpus.py --supplement fineweb` (same streaming batch-encode pattern as
TinyStories) into `data/tokenized/hf_bpe_16k/supplement_fineweb.bin`.

**Sizing:** User chose the ~1B-token option (target_bytes ≈ 3.6GiB) over ~300M/~500M-token
alternatives — actual yield: 3,844,116,015 chars / 808,365 docs / **992,803,683 tokens**
(close to the 1B target; fertility ~1.5 tok/word on this domain, consistent with D-014's
tokenizer study numbers). Combined fresh-token pool is now 17.67M (books+dict) + 520.5M
(tinystories) + 992.8M (fineweb) ≈ **1.53B tokens**; at Muennighoff's ~4-epoch repetition
ceiling that's ~6.1B tokens available against the L-tier's ~2.1B need (D-015) — comfortable
~2.9x margin, well past the "zero margin" state D-015 flagged before this data landed.
**Options considered:** ~300M tokens (fresh pool ~838M, 4-epoch ceiling ~3.35B — still clears
2.1B but thinner margin) / ~500M tokens (recommended default, ~1.06B fresh pool, ~4.2B
ceiling, 2x margin) / ~1B tokens (chosen — largest download of the three, but margin isn't
purely wasted: more raw diversity means fewer repeats are needed to hit any given token count,
which per Muennighoff is where repetition really starts hurting).
**Why:** User's call after reviewing the three sizes' tradeoffs (download/disk cost vs.
margin/diversity) via the phase-4 FineWeb-Edu decision point RW-1 always required.
**Impacts:** RW-1's tokenization work is now done (both supplements tokenized, verified via
decoded doc-boundary spot checks — no mid-document EOT tokens in either shard). RW-1's last
step, pushing these bins to the R2 bucket via `scripts/cloud/data_push.sh`, is still blocked on
RW-3 (cloud accounts: rclone isn't installed, no `r2` remote configured yet) — not attempted
this session, since the user explicitly deferred RW-3 to prioritize this data-prep thread.
**Revisit if:** the phase-4 loader's actual per-source mixing weights (RW-4's domain-mix
design, or plain fluency-focused defaults) show FineWeb-Edu's share needs to be much larger or
smaller than what a straightforward "add margin" read of this entry assumed.

## D-021 — S-tier baseline hyperparameters: lr 1e-3, effective batch ~64K tokens, eval every 100 steps  (2026-07-11, phase 4)
**Decision:** `configs/train_s_baseline.yaml` (the `p4_s_baseline` reference run): peak
**lr 1e-3**, linear warmup 30 steps (~2%) then cosine decay to `lr * 0.1`, AdamW
`betas=(0.9, 0.95)`, `weight_decay=0.1` (matrix weights only, see below), `grad_clip=1.0`.
**Effective batch ~65,536 tokens/step** (`micro_batch=16 * grad_accum=8 * seq_len=512`),
`max_steps=1500` (~98.3M tokens). **Eval every 100 steps**, 32 fixed batches (batch_size 16).
AdamW uses two param groups: weight decay on attention/FFN projection weights only, none on
norm gains or the (tied) token embedding.

**Options considered (lr):** the phase-4 spec's own draft suggested ~3e-4 (a GPT-2-small-scale
convention). Two independent, more size-appropriate estimates disagreed: nanoGPT's own
"shakespeare-char" reference config (~10.65M params, essentially the same scale as our 9.71M
S-tier) uses `lr=1e-3`; the GPT-3 paper's empirical lr-vs-log(params) fit, extrapolated to
9.71M params, also gives ~1e-3. Chose **1e-3**, matching both.

**Options considered (effective batch):** the spec's draft suggested ~0.25-0.5M tokens (GPT-2-
small's own convention). But our 17.67M-token S-tier train corpus means a 250K-500K batch
would give only ~400-800 optimizer steps over the ~100M-token baseline run -- too coarse to
resolve a cosine schedule or a clean lr-sweep divergence. nanoGPT's tiny-scale reference config
(again, ~10.65M params) uses only ~16K tokens/batch. Chose **~64K tokens** (micro_batch=16,
grad_accum=8) as a middle point matching that scale's convention while keeping `micro_batch=16`
(see D-022, MPS calibration) rather than nanoGPT's literal 64 sequences x 256 tokens.

**Why (param groups, no decay on norms/embeddings):** weight decay's "shrink toward zero"
regularization doesn't make sense for a norm gain (a single learned scale, not a projection
trading off against overfitting) or for an embedding table (decaying a rarely-seen token's row
toward zero destroys its already data-starved representation rather than regularizing it). This
model has no biases (`bias=False` throughout, D-016), so the no-decay group is exactly
`{tok_emb, all norm weights}`.

**Why (eval cadence):** S-tier steps are cheap (seconds), so frequent eval costs little and
buys fine-grained loss-curve resolution for teaching; 32 fixed batches against `val.bin`'s
179,655 tokens (~350 non-overlapping 512-token windows) gives a stable estimate without eval
dominating wall-clock.

**Revisit if:** the lr-sweep (`p4_s_lr_sweep`, this session's overnight pipeline) finds a
clearly better lr than 1e-3 within the swept range (3e-4 to 3e-3) -- see D-024's provisional,
pending-ratification note on the auto-picked lr actually used for tonight's baseline run.

## D-022 — Real measured MPS throughput for the S-tier model is flat (~11K tok/s), not D-008's ~20.8K; kept micro_batch=16 anyway  (2026-07-11, phase 4)
**Decision:** `scripts/find_batch_size.py` (D-018's calibration tool) measured the *actual*
S-tier model (9.71M params, RoPE + SwiGLU + GQA-capable attention via SDPA) on this Mac at
seq_len=512: tokens/sec is essentially **flat at ~11,000-11,800** from `micro_batch=1` through
32 (confirmed with a manual wider sweep: 1->11,563, 2->11,173, 4->11,101, 8->11,340, 16->11,198,
32->10,926 tok/s) -- barely half of D-008's dummy-TinyGPT bench (~20,800 tok/s at micro_batch=8,
same seq_len). Kept **`micro_batch=16`** in the train configs anyway (not the raw sweep's
top-throughput `micro_batch=1`), because with throughput flat, a larger micro-batch is free and
strictly reduces the number of `grad_accum` iterations needed for the same ~64K-token effective
batch (D-021) -- fewer Python-loop/data-sampling iterations per optimizer step, and it matches
the nanoGPT-tiny-scale convention already chosen for lr/batch sizing.

**Also fixed while building the calibration tool:** `find_batch_size.py`'s plateau-detection had
a classic Python bug -- `plateaued = results and tps < ...` returns the `results` list object
itself (not a bool) when `results` is empty, so `plateaued` aliased the *same mutable list*;
the very next line's `results.append(...)` then mutated that aliased object too, making the
"no prior data yet" check look non-empty by the time `if plateaued:` ran, causing the sweep to
falsely stop after just one micro-batch size every time. Fixed with an explicit `bool(results)`.
Caught by manually re-deriving the sweep's expected trace rather than trusting the first run's
one-line-and-done output.

**Why (the gap from D-008):** D-008's own numbers already flagged that "kernel-launch overhead
and unified-memory traffic dominate over raw compute" at this parameter scale; the real model's
extra fixed-cost operations per layer (RoPE cos/sin, SwiGLU's 3 matrices vs. a plain 2-matrix
GELU MLP, GQA-shaped reshapes into SDPA) plausibly push that fixed overhead higher than the
simpler dummy benchmark's, while compute itself stays tiny at either scale -- consistent with
throughput being flat rather than compute-bound-scaling with batch size.

**Revisit if:** a later tier (M/L) or a rented CUDA GPU shows throughput actually scaling with
micro-batch (i.e., this Mac's flatness is scale/hardware-specific, not a property of the model
architecture) -- re-run `find_batch_size.py` per D-018's "once per new hardware" rule regardless.

## D-023 — Two trainer bugs found via a real kill+resume test (not just the unit test): wandb swallows SIGINT; step-checkpointing off-by-one  (2026-07-11, phase 4)
**Decision:** Fixed both in `src/llmlab/train/trainer.py`, verified by actually killing and
resuming a real CLI run (`20260711_p4_resume-test`) rather than trusting `tests/test_trainer.py`
alone.

1. **`wandb.init()` installs its own SIGINT handler**, silently swallowing a plain `kill -INT`
   (i.e. Ctrl-C) so `Trainer.fit()`'s `except KeyboardInterrupt` never fired -- confirmed with
   a minimal repro (`wandb.init(mode="disabled")` + a bare `time.sleep` loop ignored `kill -INT`
   entirely; adding `signal.signal(signal.SIGINT, signal.default_int_handler)` after `wandb.init`
   fixed it). Fix: `Trainer.__init__` now reinstalls the default SIGINT handler immediately
   after `wandb.init()`.
2. **Off-by-one in what gets checkpointed as "the current step."** The original `fit()` used
   `for self.step in pbar` (`pbar` over `range(self.step, max_steps)`), so `self.step` was
   simultaneously "the step index currently executing" and "the value saved on checkpoint" --
   but Ctrl-C lands *after* a step's full body (including its `self.step`-keyed logging) has
   run, while the for-loop hasn't yet advanced its own loop variable to the next value. The
   checkpoint therefore recorded the *just-completed* step, and resume re-executed (and
   re-applied the gradient update for) that same step on top of a model that had already taken
   it once. Caught because the replayed step's logged `train_loss` (9.400) didn't match the
   original run's (9.706) for the identical step index -- only possible if the model had already
   moved, since the loader is stateless given `(seed, step)` (loader.py) and should reproduce
   identical batches. Fix: `fit()` now uses a local `step` loop variable for lr/data-indexing
   and only bumps `self.step = step + 1` (the correct "next step to run" / safe checkpoint
   value) after that step's eval/log/sample work completes.

**Verified:** after both fixes, killing `20260711_p4_resume-test` mid-run and resuming via
`scripts/train.py --resume` reproduced every subsequent logged `train_loss` bit-for-bit against
an uninterrupted control run (`20260711_p4_cpu-canary`) -- steps 0, 2, 4, 5, 6, 8 all matched to
the last decimal. Full account in that run's `notes.md`.

**Why this matters beyond just this bug:** neither issue would have been caught by
`tests/test_trainer.py`'s resume test alone -- that test manages its own `step` variable
correctly *by construction* (it doesn't drive `Trainer.fit()`), and it never sends a real
signal. This is a concrete instance of CLAUDE.md's "verify on a real run" instinct catching
something a green unit-test suite alone did not.

**Impacts:** none outside `trainer.py` -- no real training data has been produced with the
buggy resume path yet (only this session's own smoke/canary/resume-test runs), so nothing needs
retroactive correction.

## D-024 — Overnight automation: lr-sweep -> auto-pick winner -> baseline, unattended (2026-07-11, phase 4)
**Decision:** Per the user's explicit request (going to sleep, wanted zero further approval
prompts), built `scripts/orchestrate_p4_lr_sweep_and_baseline.py`: runs the 3
`p4_s_lr_sweep` configs sequentially, disqualifies any run whose `train_loss` ever goes
non-finite (NaN/Inf -- the lr-hi candidate's "watch divergence on purpose" case), picks the
survivor with the lowest logged `val_loss`, writes that lr into a **new**
`configs/train_s_baseline_auto.yaml` (D-021's own `train_s_baseline.yaml` is left untouched),
and launches the full 1500-step baseline with it. Launched via
`nohup caffeinate -dims .venv/bin/python scripts/orchestrate_...py > logs/... 2>&1 < /dev/null &
disown` -- `caffeinate -dims` prevents idle/display/disk/system sleep for as long as the
pipeline runs (critical: an M4 Mac sleeping overnight would pause/kill MPS training), and
`nohup`+`disown` detach the process tree from the shell so it survives the terminal (and this
conversation) closing.

**This is a provisional automation, not a ratified decision.** The winning lr replaces D-021's
default for *this one baseline run only* -- it has not been reviewed against the actual
sweep curves yet. Each run gets an auto-written `notes.md`; the baseline run's records which lr
won and flags itself for next-session review. Treat the resulting `20260711_p4_s-baseline`
(or `-auto` if a name collision occurred) as provisional until `notebooks/05_compare_runs.ipynb`
section 4 has been reviewed and this entry (or a superseding one) confirms or overrides the
auto-picked lr.

**Options considered:** running the lr sweep now and leaving the ~2.5h baseline for later
(rejected -- user explicitly wanted the full chain unattended overnight); shrinking the
baseline's step count to fit the session (rejected for the same reason: user wants the real
1500-step reference run, not a shortened stand-in).

**Why:** ~10,700 tok/s measured throughput (D-022) means the full baseline (~98.3M tokens) takes
~2.5h and the 3 sweep runs ~30min apiece (~1.5h total) -- ~4h combined, past what's reasonable
to ask the user to stay awake for, but well suited to overnight unattended compute per CLAUDE.md's
"Python scripts run from terminal" rule for long jobs.

**Revisit if:** the pipeline's chosen lr conflicts with a more careful reading of the sweep
curves next session (e.g. the winner only "won" due to short-run noise, not a real trend) --
override it and log a new D-entry naming this one, per the change-management protocol.

## D-025 — Overnight lr sweep result: D-021's lr=1e-3 ratified (not overridden); p4_s_baseline is complete  (2026-07-12, phase 4)
**Decision:** Reviewed the overnight pipeline's (D-024) 3-way lr sweep against equal-step val_loss
curves rather than just the final numbers. lr=1e-3 (`20260711_p4_s-lr-sweep-mid`) was **strictly
ahead of both lr=3e-4 (`-lo`) and lr=3e-3 (`-hi`) at every logged checkpoint** (steps 0/50/.../250),
not just at the end -- val_loss 4.729 vs 5.249 (lo) and 4.843 (hi). This **ratifies D-021's
original default**, it does not override it: the automation's "provisional, pending review" flag
is resolved with no change needed. `p4_s_baseline` (1500 steps, lr=1e-3, final val_loss 3.5037 /
ppl 33.2) is therefore the real, final S-tier reference run -- not a placeholder.

**Additional finding (lo/hi behavior):** lr=3e-4 was undertrained rather than unstable (smaller
per-step movement, not a quality problem, just needs more steps). lr=3e-3 did not diverge
(`grad_clip=1.0` held) but was still clearly worse than 1e-3, *despite* ending with a lower mean
grad_norm (0.566) than 1e-3's own run (0.687) -- i.e. `grad_clip` bounds the damage from too
large an lr, it does not rescue the outcome. "Didn't diverge" is not evidence of "was a good lr."

**Why:** the phase-4 exit criteria (`docs/phases/phase4_training.md`) require the baseline S run
"finished & registered" with samples reading English-ish and resume verified -- all now true:
baseline registered with a real verdict (not "review and fill in notes.md"), samples show fluent
prose picking up the corpus's Socratic-dialogue register by step 800 (see the run's notes.md),
and resume was verified for real in this session (D-023) with two genuine bugs fixed along the
way. `notebooks/05_compare_runs.ipynb` renders all of the above cleanly.

**Impacts:** Phase 4 checklist item 4 (First experiments) is now fully done. Milestone M1
(per the phase's exit criteria) can be declared -- see PROGRESS.md.
**Revisit if:** phase 5's noise-floor runs (3 seeds of this same baseline) show the sweep's
margins were within seed noise after all -- unlikely given the consistency across every
checkpoint, but that's exactly what the noise-floor protocol (`docs/EXPERIMENTS.md`) is for.

## D-026 — R2 credentials via `.env` + env-var rclone remote (no config file); rclone installed from official zip, not Homebrew  (2026-07-12, phase 4 / RW-3)
**Decision:** User created the Cloudflare R2 bucket (`llmlab`) this session, prompting the first
real step of RW-3. Two sub-decisions:
1. **Credentials live in a root `.env`** (gitignored — repo is public), not in
   `~/.config/rclone/rclone.conf` via the interactive `rclone config` wizard. The five
   `RCLONE_CONFIG_R2_*` env vars (`TYPE`, `PROVIDER`, `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`,
   `ENDPOINT`) configure rclone's `r2:` remote directly — this is the *same* mechanism
   `docker/entrypoint.sh` already used for the pod template, so local Mac and cloud pod now
   share one credential format instead of two. `.env.example` (committed) documents where to
   get each value; `scripts/cloud/data_push.sh` sources `.env` automatically if present.
2. **rclone installed from the official release zip** (`downloads.rclone.org/rclone-current-osx-arm64.zip`
   → `~/bin/rclone`, `~/bin` added to `PATH` in `.zshrc`), not `brew install rclone` — Homebrew
   on this machine is currently broken (`/opt/homebrew` ownership issue *and* a Ruby version-check
   crash on macOS 26.5.2, which Homebrew doesn't yet recognize). Did not run `sudo chown` to fix
   Homebrew, since that's a system-wide permissions change outside this session's scope — flagging
   here in case the user wants to fix Homebrew properly later; the zip install is a full
   substitute for our purposes (single static binary, no other deps needed).

**Why:** the user asked specifically for credentials in `.env` (not Homebrew's config file) and
for the file to be excluded from git since the repo is public. Reusing the pod template's
existing `RCLONE_CONFIG_R2_*` var names (already documented in `docker/entrypoint.sh` for
Docker fast-start) avoids inventing a second credential format.

**Impacts:** `docs/CLOUD.md` step 4 rewritten to match; `.gitignore` gained `.env`;
`scripts/cloud/data_push.sh` now sources `.env`. RW-1's last remaining step (push tokenized bins
to R2) is now complete — see below.
**Revisit if:** Homebrew gets fixed later (e.g. a Homebrew release adds support for this macOS
version) — fine to switch to `brew install rclone` then, no functional difference either way.

**Follow-up same day:** the bucket is actually named `llm` (not `llmlab` as first assumed/written
into docs/scripts) — the mismatch caused an initial `403 AccessDenied` from R2 (misread as a
credentials problem; it was a wrong bucket name in the `r2:llmlab` path, not a permissions issue
in the token). Corrected in `.env`, `.env.example`, `docs/CLOUD.md`, and `docker/entrypoint.sh`'s
comment. `scripts/cloud/data_push.sh` then ran successfully: `data/tokenized/` (2.879 GiB, 16
files — train/val, both supplements + docstarts, meta.json, all 3 tokenizer vocab sets) is now
in `r2:llm/data/tokenized/`, verified via `rclone lsf -R`. RW-1 and this R2 sub-step of RW-3 are
both fully done.

## D-027 — Cloud provider: gpuhub chosen over RunPod (~2x cheaper), native image-snapshot workflow, RunPod kept as documented-but-unbuilt fallback  (2026-07-12, RW-3)
**Decision:** **gpuhub** is the active cloud provider for burst training (RTX 5090 tier for
M/L-tier confirmations and the phase-9 hero run, per D-010's hybrid-compute strategy — that
strategy itself is unchanged, only the provider choice within it). User's own quoted pricing:
RTX 5090 **$0.46/hr on gpuhub vs $0.99/hr on RunPod**; RTX PRO 6000 **$0.91/hr vs $1.99/hr** —
gpuhub is ~50% cheaper on both tiers checked, same hardware class either way. User purchased $10
gpuhub credit and installed Docker Desktop locally this session.

**Docker workflow — option (a) from `docs/CLOUD_GPUHUB.md`'s "Open decision":** fully adapt to
gpuhub's native flow (rent from their pre-built PyTorch/CUDA catalog → run setup script live over
SSH → "Save Image" snapshots the configured disk for reuse), NOT the Docker-Hub-push flow gpuhub
doesn't support (see D-017/`docs/CLOUD_GPUHUB.md` §1 for why: gpuhub refuses to pull images from
any third-party registry). `docker/Dockerfile`'s `RUN` lines remain the source of truth for
*what* needs installing — a new gpuhub-native setup script (`scripts/cloud/gpuhub_setup.sh`)
translates them into shell commands run on the live instance instead of `docker build` steps.

**Options considered:** (a) gpuhub-native, build once now — chosen; (b) stay RunPod-only,
keep the existing Docker-Hub plan as the *only* plan — rejected, costs 2x forever for the same
hardware; (c) build both providers' flows now in parallel — rejected as premature effort before
a single successful cloud run has happened on either.
**Why:** Pure cost at fixed hardware quality; the one-time cost of learning gpuhub's flow
(this session's research + first dry run) is paid once and amortized over every future rental.
**Impacts:** RW-3's Docker Hub sub-step is superseded for the *active* path — no image gets
built/pushed against gpuhub right now. `docs/CLOUD.md` (RunPod/Docker-Hub flow) is kept fully
intact and unmodified as a documented fallback, explicitly not worked on further right now per
the user's instruction. `docs/CLOUD_GPUHUB.md` is the live, actively-maintained playbook going
forward — its "Open decision" section is now resolved by this entry. PROGRESS.md's RW-3 row
updated accordingly.
**Revisit if:** gpuhub pricing/availability changes materially, gpuhub's Save-Image/native flow
proves unworkable in practice once actually exercised (fall back to RunPod's already-documented
Docker Hub flow, which needs no rework since it was never abandoned, just deprioritized), or a
future phase needs a provider feature gpuhub lacks (e.g. genuine multi-node distributed training,
which gpuhub doesn't support on non-A100 cards per the research).

## D-028 — First gpuhub dry run on RTX 4080 Super (Ada/sm_89), not RTX 5090 directly  (2026-07-12, RW-3)
**Decision:** Validate the whole gpuhub pipeline (SSH, `gpuhub_setup.sh`, R2 data pull, deps
install, a training smoke run, Save Image) on gpuhub's **RTX 4080 Super** tier ($0.25/hr) before
touching the target RTX 5090. Immediate trigger: RTX 5090 had zero inventory available at rental
time; RTX PRO 6000 was available but offered no compatibility advantage (same Blackwell/sm_120
family, same CUDA≥12.8/PyTorch≥2.7.1 requirement) at ~4x the price.
**Options considered:** RTX 5090 (unavailable) / RTX PRO 6000 ($0.91/hr, Blackwell, available but
pricier with no upside for this purpose) / **RTX 4080 Super ($0.25/hr, chosen)** — Ada Lovelace
(sm_89), a long-established architecture with no CUDA≥12.8/PyTorch-nightly wrinkles.
**Why:** cheapest available option AND removes a variable — Blackwell's strict version
requirements (flagged in `docs/CLOUD_GPUHUB.md` §2) are exactly the kind of thing you don't want
to be debugging at the same time as "does SSH/rclone/git-clone even work on this platform."
Validate the pipeline on boring, well-supported hardware first; the eventual 5090 rental then
only needs to re-verify the CUDA-version/driver side (`ldconfig -p | grep cuda`, not `nvidia-smi`
— the instance listing's displayed "CUDA: 13.2" is presumed to be the driver's max-supported
version per that same gotcha, not the installed toolkit version; confirm once SSH'd in).
**Impacts:** none to the training code (already fully device-agnostic per CLOUD.md's portability
rules) — this only affects which instance `scripts/cloud/gpuhub_setup.sh` gets exercised on
first. A saved image built on the 4080 instance should still be usable as a starting point for a
future 5090 rental (Save Image snapshots the system disk, not GPU-specific state), though the
Blackwell CUDA/PyTorch requirement may still need a nightly-PyTorch bump at that point if the
saved image's PyTorch predates 2.7.1.
**Revisit if:** something in the pipeline turns out to be GPU-architecture-sensitive in a way
that doesn't transfer from Ada to Blackwell (unlikely — training code only touches
`get_device()`/`autocast_ctx()`, no architecture-specific branches).

## D-029 — gpuhub cloud pipeline validated live end-to-end on RTX 4080; 3 real setup bugs found and fixed  (2026-07-12, RW-3)
**Decision:** `scripts/cloud/gpuhub_setup.sh` (D-027/D-028's planned artifact) was executed for
real via SSH against a rented gpuhub RTX 4080 Super instance, not just written from docs. Result:
full pipeline passes — see `experiments/20260712_p4_s-smoke_cloud4080/notes.md` for the
registered run (`experiments/registry.csv`). Three real bugs surfaced only by actually running
it, none of which the docs research (D-027's `docs/CLOUD_GPUHUB.md`) predicted, all now fixed in
`gpuhub_setup.sh`:
1. **conda's `python`/`pip` are not on `PATH` in a non-interactive SSH session** — no `conda
   init` has been run on a fresh catalog image, so `remote_setup.sh`'s bare `python`/`pip` calls
   would silently fail with "command not found." Fixed: script now exports
   `PATH=/root/miniconda3/bin:$PATH` and persists it to `.bashrc` for later interactive/tmux use.
2. **`rclone` is not preinstalled** on the PyTorch 2.8.0/CUDA 12.8 catalog image (only `tmux` gets
   auto-installed, by `remote_setup.sh`'s existing apt check). Fixed: script now installs rclone
   via the official install script if missing, matching `docker/Dockerfile`'s approach.
3. **Data-disk mount confirmed live as `/root/autodl-tmp`** — resolves the docs'
   self-inconsistent `autodl-tmp` vs `gpuhub-tmp` naming (`docs/CLOUD_GPUHUB.md`'s provenance
   note) with a real answer on real hardware; the script's live-detection logic (checks both,
   picks whichever exists) worked as designed and needed no change.

**Also confirmed live, matching the docs-based research exactly:**
`nvidia-smi` reported "CUDA Version: 13.2" (the instance-picker UI showed this too) while
`torch.version.cuda` was actually `'12.8'` — the driver-ceiling-vs-installed-runtime gotcha the
docs warned about, now empirically verified rather than just documented as a warning.

**Measured throughput: 99,554 tok/s** on the S-tier model (9.71M params) — ~8.5x D-022's Mac MPS
number (~11,000-11,800 tok/s), on the RTX 4080 dry-run tier, not even the target RTX 5090.
Sample quality/loss trajectory (train_loss 9.69→5.39, val_loss 9.44→5.26) matched the original
Mac smoke run almost exactly, confirming numerical correctness of the port, not just "it ran."

**Also verified: CUDA-trained checkpoint loads on Mac MPS.** `latest.pt` (saved on the gpuhub
CUDA instance) was `scp`'d back and loaded via `torch.load(..., map_location=get_device())` on
the Mac — confirms `docs/CLOUD.md`'s cross-device-checkpoint portability rule works in practice,
not just in the code's design.

**Why this matters beyond the fix:** this is the same lesson as D-022/D-023 — docs/design review
finds real issues, but only actually running the thing on the real target finds the rest.
Neither the 33-page gpuhub docs research (D-027) nor code review would have caught the missing
`PATH` or missing `rclone`, since neither is documented anywhere in gpuhub's docs.
**Impacts:** `scripts/cloud/gpuhub_setup.sh` updated in place (not a new decision to revisit —
this entry documents *why* those two blocks are in the script, since they'd look like unexplained
defensive code otherwise). RW-3 effectively complete for the RTX-4080-validated path; only
remaining step is repeating the CUDA-version check on an actual RTX 5090 once inventory allows,
per D-028.
**Revisit if:** a future gpuhub catalog image ships with `conda init` already run or `rclone`
preinstalled — the script's guards are no-ops in that case, harmless either way.

## D-030 — RTX 4080 capacity measured across tiers/seq_len; GPU-specific sweet-spot micro_batch differs from Mac's; L-tier hero run could plausibly cost ~$3-4 on this tier alone  (2026-07-12, RW-3)
**Decision:** Ran `scripts/find_batch_size.py` (D-018) live on the gpuhub RTX 4080 Super
instance across all three model tiers and three sequence lengths, plus an isolated
confirmation script to rule out measurement artifacts. Full writeup with methodology and
reasoning: `docs/learnings/20260712_gpuhub-rtx4080-capacity.md` (this entry is the terse
decision-log version; that doc is the teaching version).

**Sweet-spot tok/s (seq_len=512):** S-tier (9.71M) micro_batch=32 → 198,088 tok/s; M-tier
(34.62M) micro_batch=32 → 72,611 tok/s; L-tier (104.80M) micro_batch=16 → 42,499 tok/s. These
are DIFFERENT sweet-spot micro-batches than D-022's Mac-derived defaults currently sitting in
`configs/train_s_*.yaml` (`micro_batch=16`) — D-018's own rule ("recalibrate per hardware, never
assume") applies here: **update micro_batch to 32 before a real S-tier run on this GPU tier**,
grad_accum re-factored to keep the same effective-batch hyperparameter.

**Non-obvious finding:** throughput does NOT plateau flat past the sweet spot on this hardware —
it *regresses* (S-tier: 198K→132K→87K tok/s at micro_batch 32→64→128, confirmed twice plus an
isolated single-batch check with explicit cache-clearing to rule out cross-sweep memory
fragmentation as a confound). Memory itself scales cleanly linearly (5.87→11.66→23.25GB) — it's
specifically compute throughput that regresses past the sweet spot, a different pattern from
D-008's Mac "cliff" (that was a sudden collapse; this is a gradual real regression, no cliff,
until an actual OOM at micro_batch=256, ~31GB).

**Seq-len scaling:** the sweet spot holds at a roughly constant ~16,384 tokens-per-forward-pass
regardless of how that's split between batch and sequence length (512×32 ≈ 1024×16 ≈ 2048×8, all
~192-198K tok/s) — going to a longer default context costs no real throughput on this hardware,
it just needs a smaller micro-batch to stay at the sweet spot.

**Cost projection (validated, not blind extrapolation — see reasoning below):** S-tier ablation
run (75M tokens) ≈ 6.3 min ≈ $0.03; M-tier confirmation (1B tokens, illustrative) ≈ 3.8hr ≈
$0.96; **L-tier hero run (2.1B tokens, D-015's Chinchilla budget) ≈ 13.7hr ≈ $3.43** — using the
sweet-spot micro-batch, not the current Mac-tuned default. This projection is grounded, not
speculative: the raw fwd+bwd-only benchmark at micro_batch=16 (98,757 tok/s) matched the actual
full training run's measured throughput (99,554 tok/s, `20260712_p4_s-smoke_cloud4080`) almost
exactly, meaning optimizer/data-loading/eval/logging overhead is negligible on this hardware, so
the sweep numbers can be trusted as real achievable training throughput, not just idealized
compute-only numbers.

**Options considered:** trust the instance-listing spec sheet (rejected — matches this project's
already-learned D-008 lesson that spec sheets don't predict real throughput/cliffs); measure only
S-tier (rejected — the whole point was informing which future *tiers* of run this cheap
instance can handle); skip the isolated-confirmation check on the throughput regression
(rejected — a repeatable-but-unexplained regression needed at least one methodology-artifact
check before trusting it as real).

**Why this matters:** D-010 originally planned the RTX 5090 as the burst option specifically to
fix D-008's "1.5-3 weeks on Mac" hero-run timeline problem, budgeted at "$10-20 overnight." This
finding suggests the *cheaper* dry-run tier alone could plausibly finish the same hero run in
under 14 hours for roughly a third of that budget — worth taking seriously as more than a
sandbox, though a real (not synthetic-benchmark) M/L-tier validation run should happen before
committing hours at those tiers, and the RTX 5090 plan isn't retired (a short real run there is
still warranted once inventory allows, per D-028).

**Also flagged (not fixed, not blocking):** (1) `find_batch_size.py`'s reported `mem_gb` column
is unreliable (stayed flat across micro-batch sizes within a tier, contradicting the isolated
check's real linear-scaling numbers) — likely reading instantaneous rather than peak CUDA memory;
worth fixing in a future session, not urgent since the isolated check gave trustworthy numbers.
(2) `GPT.forward()` hard-rejects sequences longer than `model_config.max_seq_len`, which will
block phase 5 Wave B's planned RoPE-extrapolation probe (train at 512, eval at 1024/2048) until
that guard is relaxed for eval-only forward passes — discovered incidentally while benchmarking
seq_len scaling, not yet fixed, flagged for whoever starts Wave B.
**Impacts:** none to code (measurement only). `configs/train_s_*.yaml` should get
`micro_batch=32` before the next real cloud run on this GPU tier (not changed in this entry —
that's a config edit for whenever the next real run is launched, to avoid touching settled
configs speculatively).
**Revisit if:** a real (non-synthetic) M/L-tier training run on this hardware shows throughput
meaningfully different from these fwd+bwd-only sweep numbers — recalibrate the cost projection
table above if so.

## D-031 — RTX 4080 capacity matrix completed: M/L-tier seq_len scaling confirms the constant-tokens-per-step finding generalizes  (2026-07-12, RW-3)
**Decision:** Extended D-030's S-tier-only seq_len sweep to M and L tiers (before switching to a
rented RTX 5090 for the same measurement — this data becomes unrepeatable once the RTX 4080
instance is stopped, and it's a few cents of GPU time). Full matrix, sweet-spot micro_batch only:

| Tier | seq_len=512 | seq_len=1024 | seq_len=2048 | Sweet-spot tokens/step |
|---|---|---|---|---|
| S (9.71M) | mb=32 → 198,088 tok/s | mb=16 → 198,406 tok/s | mb=8 → 191,693 tok/s | 16,384 |
| M (34.62M) | mb=32 → 72,611 tok/s | mb=16 → 70,695 tok/s | mb=8 → 72,535 tok/s | 16,384 |
| L (104.80M) | mb=16 → 42,499 tok/s | mb=8 → 42,655 tok/s | mb=4 → 40,450 tok/s | 8,192 |

**Finding confirmed, not just an S-tier coincidence:** each tier has its own fixed "sweet-spot
tokens-per-forward-pass" constant (S and M both ~16,384; L ~8,192 — half, consistent with L's
higher per-token compute cost saturating the GPU at a smaller batch×seq_len product). Within a
tier, tok/s stays flat across all three sequence lengths as long as micro_batch is halved each
time seq_len doubles. **Practical implication: a longer default context window costs ~nothing in
total throughput on this hardware** at these model sizes — the real constraint is the
tokens-per-step "work packet," not sequence length per se.
**Why:** answers "what about 1024/2048" for M/L tiers specifically, and confirms (rather than
assumes) that D-030's S-tier pattern generalizes — this project's repeated lesson (D-008, D-018,
D-022) is that hardware behavior at one scale doesn't automatically transfer to another without
checking.
**Impacts:** `docs/learnings/20260712_gpuhub-rtx4080-capacity.md` updated with the full matrix.
No config changes (measurement only, same as D-030).
**Revisit if:** the equivalent RTX 5090 sweep (planned next, same session) shows a qualitatively
different pattern — e.g., if the 5090's larger compute headroom makes it NOT saturate the same
way, the "sweet-spot tokens/step is roughly hardware+tier-constant" framing would need revising
per-GPU, not treated as a general law.

## D-032 — RTX 5090 comparison sweep: strictly better than the RTX 4080 tier, not just faster — recommend defaulting to 5090 for all real runs when available  (2026-07-12, RW-3)
**Decision:** Ran the identical 9-sweep matrix (D-030/D-031's methodology: 3 tiers × 3 seq_lens,
`find_batch_size.py`) on a real RTX 5090 instance ($0.46/hr), triggering D-031's own "revisit if"
condition. Result: the 4080's throughput-regression-past-sweet-spot pattern did **not** reproduce
— 5090 sweeps mostly kept climbing to the tested ceiling (S-tier @512 hit 627,326 tok/s at
micro_batch=128, still rising, untested beyond that) or hit a real CUDA OOM, never the gradual
regression seen on the 4080. This confirms the 4080's regression is a quirk of that specific card
(plausibly related to it being a modified/non-stock 32GB-VRAM part — see D-030), not a property
of these model sizes in general.

**The "sweet-spot tokens-per-step is roughly tier-constant" finding (D-031) held on the 5090
too**, at a higher constant: S-tier ~65,536 tokens/step (4x the 4080's ~16,384) — 627,326 /
607,058 / 569,295 tok/s at 512/1024/2048. L-tier ~16,384 tokens/step (numerically matching the
4080's S/M constant — coincidence, not a cross-GPU law) — 127,033 / 122,598 / 114,984 tok/s.
(M-tier's 512-seq_len sweep stopped early at the plateau-tolerance threshold with mb=64 only 0.6%
behind mb=32 — likely an underestimate of its true ceiling, unlike the cleaner S/L data.)

**Head-to-head cost comparison (sweet-spot tok/s, seq_len=512):**

| Task | 4080 ($0.25/hr) | 5090 ($0.46/hr) | Speedup | Cheaper by |
|---|---|---|---|---|
| S-tier ablation (75M tok) | 0.11hr / $0.03 | 0.03hr / $0.02 | 3.17x | $0.01 |
| M-tier (1B tok, illustrative) | 3.83hr / $0.96 | 1.28hr / $0.59 | 2.98x | $0.37 |
| L-tier hero run (2.1B tok, D-015) | 13.73hr / $3.43 | 4.59hr / $2.11 | 2.99x | $1.32 |

**The 5090 is strictly better across every tier tested** — despite costing 84% more per hour, it's
~3x faster, making it BOTH faster and cheaper per completed run. This isn't a "pay more for
speed" tradeoff; it's a genuine free lunch at these model sizes. **Recommendation: default to
RTX 5090 whenever gpuhub has inventory; treat the RTX 4080 tier as a near-free dry-run/debugging
sandbox only** (a smoke test costs about a penny on either GPU, so the 4080's lower hourly rate
doesn't matter for that use case — it matters for real runs, where the 5090 wins outright).

**Options considered:** keep favoring the cheap 4080 tier per D-030's initial framing (rejected —
that framing was written before a real 5090 comparison existed, explicitly flagged as
provisional pending this exact measurement); split by tier (e.g. 4080 for S-tier ablations, 5090
for M/L) — rejected, the 5090 wins even at S-tier, no tier favors the 4080 once you account for
$/run rather than $/hr.

**Also: a process bug found while setting up this comparison, unrelated to either GPU.** Setting
up the 5090 instance via the `curl`-from-GitHub one-liner (`docs/CLOUD_GPUHUB.md` §11) reproduced
D-029's exact "python/pip not on PATH" bug — because D-029's fix to `scripts/cloud/gpuhub_setup.sh`
was made locally and never committed/pushed, so the GitHub-hosted copy the curl one-liner fetches
was still the broken version. Worked around by `scp`-ing the local fixed copy directly (same
method as the very first setup). **Lesson: an uncommitted fix to a script designed to be
fetched by URL isn't actually fixed for that workflow** — logged here rather than silently
patched, per this project's habit of recording process lessons alongside technical ones (see
D-022's/D-023's precedent). The fix itself still hasn't been pushed as of this entry — that's a
user call (git commits are user-initiated per CLAUDE.md), flagged for the session wrap-up.
**Impacts:** `docs/learnings/20260712_gpuhub-rtx4080-capacity.md` extended with the full
comparison (title updated to reflect both GPUs). `docs/CLOUD_GPUHUB.md` §10 updated to recommend
the 5090 by default. No training config changes yet (measurement only — real configs should be
updated with the measured sweet-spot micro_batch immediately before whichever GPU tier is
actually used for the next real run, per D-018).
**Revisit if:** gpuhub's 5090 inventory/pricing changes materially, or a real (non-synthetic)
training run on the 5090 shows throughput meaningfully different from these fwd+bwd-only sweep
numbers (same caveat as D-030 — only S-tier has a real-training calibration point so far, from
the 4080; worth a real 5090 smoke-test run before fully trusting the M/L projections).

## D-033 — RTX PRO 6000 extreme-capacity test: confirms D-018's "not worth it" prediction, AND reveals the 5090 comparison (D-032) used an inconsistent, likely-truncated sweep methodology  (2026-07-12, RW-3)
**Decision:** At the user's explicit request ("test it to the extreme with custom
configurations"), ran a 15-sweep matrix (3 tiers × 5 seq_lens: 512/1024/2048/4096/8192, vs the
prior 3-seq_len matrices) on a rented RTX PRO 6000 ($0.91/hr, 96GB VRAM), with early-stopping
disabled (`--plateau-tolerance -1`) and a high micro-batch ceiling (`--max-micro-batch 4096`) so
every sweep ran to a **real CUDA OOM**, not an early-stop heuristic or an arbitrary cap. All 120
raw data points appended to `docs/results/cloud_gpu_benchmarks.csv` (233 rows total across all
three GPUs now — full methodology and per-point data, not just sweet-spot summaries).

**Finding 1 (expected, now confirmed empirically): RTX PRO 6000 is not worth it for this
project's model sizes, exactly as D-018 predicted from VRAM-need reasoning alone.** Despite
having higher raw sweet-spot tok/s than the 5090 at every tier (S: 644,000 vs 627,326; M:
246,864 vs 216,199; L: 153,490 vs 127,033), its $0.91/hr rate (~2x the 5090's $0.46/hr) more than
cancels the throughput edge:

| Tier (budget) | RTX 4080 | RTX 5090 | RTX PRO 6000 |
|---|---|---|---|
| S (75M tok) | $0.026 | $0.015 | $0.029 |
| M (1B tok) | $0.956 | $0.591 | $1.024 |
| L (2.1B tok) | $3.431 | $2.112 | $3.458 |

RTX PRO 6000 is the *most expensive* option at every tier — worse than even the 4080 dry-run
tier. Its 96GB VRAM buys nothing our ~10-105M-param models can use (matches D-018/D-032's
VRAM-need math directly, now with a real measurement instead of just reasoning about it).

**Finding 2 (unexpected, and more important): the throughput-regression-past-sweet-spot pattern
(first seen on the RTX 4080, D-030) is NOT a 4080-specific quirk — it reproduced cleanly on the
RTX PRO 6000 too, at every tier**, once tested with the same extreme (no-early-stop, high-ceiling)
methodology. Example, S-tier @512: climbs to 644,000 tok/s at micro_batch=128, then regresses to
555,119 (mb=256) and 537,031 (mb=512) before OOM at 1024 — the same shape as the 4080's curve,
just at a ~3.3x higher absolute ceiling. M-tier and L-tier show the identical shape.

**This means D-032's "the 5090 doesn't show this regression" conclusion was premature — not
wrong about the 5090's superiority, but based on an incomplete test.** The 5090 sweep (same
session, prior turn) used `--max-micro-batch 128` (a hard cap) WITHOUT `--plateau-tolerance -1`
(so early-stopping was still active) — a materially different, more conservative methodology than
this RTX PRO 6000 run. Concretely: the 5090's M-tier@512 sweep stopped at micro_batch=64
(214,834 tok/s, interpreted as "plateaued" since it's ~0.6% below mb=32's 216,199) — but the RTX
PRO 6000's *un-capped, non-early-stopping* sweep of the same tier/seq_len kept climbing well past
that point, to a true peak of 246,864 at mb=64 before regressing. The 5090's recorded numbers are
therefore **likely a lower bound**, not its true ceiling, for the M/L tiers specifically (S-tier's
5090 numbers are probably fine — the RTX PRO 6000's true S-tier peaks landed at the *exact same*
micro-batch values the 5090's capped sweep reported: 128/64/32 at 512/1024/2048).

**Practical implication:** this doesn't change the qualitative conclusion (5090 still the best
value once available) — if anything, a fair re-test would likely make the 5090's numbers *better*,
strengthening that conclusion further, since RTX PRO 6000 already loses on cost even using the
5090's conservative/underestimated figures. But the exact 5090 M/L-tier cost numbers in D-032
should be treated as upper-bound cost estimates (i.e., real cost is probably a bit lower), not
final numbers, until re-tested with this same extreme methodology.

**Tokens-per-step sweet-spot constant (D-031's finding) held cleanly on RTX PRO 6000 too**, now
confirmed across 5 seq_lens instead of 3: S-tier ~65,536 tokens/step (matches the 5090's number
exactly); M-tier ~32,768 (a new, cleaner data point than either other GPU had shown); L-tier
~16,384, consistent across all five tested seq_lengths (512 through 8192) without exception —
the cleanest confirmation of this pattern yet.

**Options considered:** trust the RTX PRO 6000 for future big-model headroom on VRAM alone
(rejected — this project's models are nowhere near VRAM-bound at any tier through L, D-018);
skip re-testing after finding the 4080-style regression (rejected — the point of "extreme"
testing was exactly to check whether earlier conclusions held up under harder conditions, and
they didn't fully).
**Why this matters:** a repeat of this project's core lesson (D-008, D-018, D-022, D-023,
D-029) — measuring beats assuming, AND a partial measurement can itself mislead if the test
conditions differ between comparison points. The fix wasn't "measure once" but "measure the same
way every time you compare."
**Impacts:** `docs/results/cloud_gpu_benchmarks.csv` now has all three GPUs' full sweep data
(233 rows). `docs/learnings/20260712_gpuhub-rtx4080-capacity.md` to be extended with this
section. `docs/CLOUD_GPUHUB.md` §10 to get a 3-way table. RTX PRO 6000 confirmed NOT recommended
for this project going forward — RW-3/CLOUD_GPUHUB.md should default to RTX 5090, matching
D-032, with RTX PRO 6000 noted as tested-and-ruled-out rather than untested.
**Revisit if:** a future phase's model size actually becomes VRAM-bound (would need to be far
beyond L-tier's 105M params) — re-open RTX PRO 6000 as an option then. Also revisit the exact
5090 M/L-tier cost figures if/when a same-methodology 5090 re-test happens (optional, cheap,
offered to the user but not yet done as of this entry).

## D-034 — Same-methodology RTX 5090 re-test: user's architectural hypothesis confirmed (PRO 6000's edge grows with sequence length) but doesn't flip the cost conclusion  (2026-07-12, RW-3)
**Decision:** Re-ran the RTX 5090 with the exact same "extreme" methodology as D-033's RTX PRO
6000 test (`--plateau-tolerance -1`, `--max-micro-batch 4096`, every sweep run to real OOM) —
closing the methodology gap D-033 flagged. 90 new rows appended to
`docs/results/cloud_gpu_benchmarks.csv` (tagged `sweep_extreme` to distinguish from the earlier,
more conservative `sweep` source for the same GPU — both kept for the record, not overwritten).

**User's hypothesis going in: "maybe PRO 6000 only shows an advantage at longer context (2048-
4096+), not at short sequences" — confirmed, with real numbers.** PRO 6000's throughput
advantage over the 5090 grows monotonically with sequence length at every tier tested:

| Tier | seq=512 | seq=1024 | seq=2048 | seq=4096 | seq=8192 |
|---|---|---|---|---|---|
| S | +2.2% | +6.3% | +6.4% | +11.6% | +19.1% |
| M | +14.4% | +16.2% | +17.5% | +20.5% | +25.2% |
| L | +20.4% | +21.5% | +23.3% | +25.9% | +30.3% |

(% = how much faster PRO 6000's sweet-spot tok/s is than the 5090's, same tier/seq_len, both now
measured with identical extreme methodology.) This is very likely a **memory-bandwidth** effect:
longer sequences push more memory traffic per token through attention, and PRO 6000 (a bigger,
more complete Blackwell die than the consumer 5090) most plausibly has higher memory bandwidth —
a real architectural difference, not a measurement artifact, and directionally exactly what the
user predicted.

**But this does NOT flip the cost recommendation at this project's current model sizes.** Even at
the widest tested gap (L-tier @ seq_len=8192, PRO 6000 30.3% faster), the cost math still favors
the 5090: $3.14 (5090, 6.83hr) vs $4.77 (PRO 6000, 5.24hr) for the L-tier hero-run token budget.
PRO 6000's price premium (~98% more per hour) still exceeds its largest measured throughput
edge (30.3%) at every tier/seq_len combination tested. **RTX 5090 remains the right default.**

**Corrected/superseded numbers from D-032/D-033:** the extreme-methodology 5090 re-test also
revealed the earlier "sweet-spot tokens-per-step is a single sharp constant" framing (D-031)
oversimplifies the 5090 specifically — its S-tier peak is a **broad, flat plateau** (roughly
32,768-65,536 tokens/step, several candidate micro-batches within ~1% of each other) rather than
one sharp winner, unlike the 4080's and PRO 6000's noticeably sharper single-point peaks. L-tier's
constant (16,384 tokens/step) held cleanly and identically on both the 5090 and PRO 6000 across
all 5 seq_lens, though — the "constant tokens/step" finding is solid at L-tier, just noisier
right at the S-tier peak specifically.
**Options considered:** trust the D-033 flag that 5090 numbers were "a lower bound" without
re-testing (rejected — user explicitly asked to finish the extreme test, and the corrected
numbers matter for the specific architectural question raised); conclude PRO 6000 is worth it at
long context (rejected — cost math still favors 5090 even at the widest measured gap).
**Why this matters:** validates that pushing back on a measurement's completeness (as the user
did) is exactly the right instinct — the correction was real, not just methodological pedantry,
and it answered a genuine architectural question with real numbers instead of speculation.
**Impacts:** `docs/results/cloud_gpu_benchmarks.csv` now has 324 rows (all three GPUs, both 5090
methodologies preserved for the record). `docs/learnings/20260712_gpuhub-rtx4080-capacity.md`
and `docs/CLOUD_GPUHUB.md` §10 to be updated with the corrected numbers and the growing-gap
finding.
**Revisit if:** a future model size or context-length requirement pushes PRO 6000's throughput
edge past its ~98% price premium — extrapolating the L-tier trend (20.4%→30.3% over 512→8192),
that crossover looks far beyond any context length this project currently plans to use, but
worth re-checking if that assumption changes.

## D-035 — Phase 5 seed-noise floor established: std 0.0062, spread 0.0150 across 3 seeds; first real training confirmation of RTX 5090 throughput (2026-07-12, phase 5)
**Decision:** Ran phase 5's mandatory first task (`docs/phases/phase5_ablations.md` "Standing
protocol") — the S-tier baseline recipe (D-021) at 3 seeds, 1500 steps/98.3M tokens each. Reused
`20260711_p4_s-baseline` (seed=1337) as seed 1/3 rather than re-running it, and added
`20260712_p5_s-seed-{1338,1339}` for seeds 2-3 — identical config in every other respect
(`configs/train_s_seed_{1338,1339}.yaml`, only `seed`/`phase`/`baseline_run`/`variable_changed`
differ from `train_s_baseline.yaml`; micro_batch/grad_accum deliberately left at the Mac-tuned
16/8, not GPU-tuned, so seed was the only variable per the ablation protocol's rule 1).

**Result: mean val_loss 3.5043, std 0.0062, spread (max-min) 0.0150** (1337: 3.5037, 1338:
3.4970, 1339: 3.5121). Logged in `docs/EXPERIMENTS.md` as the standing rule: any later Wave A-G
verdict with a val_loss delta smaller than ~0.015-0.02 from its named baseline must be reported
as within-noise, not a real effect.

**Both seed runs executed on the already-running RTX 5090 gpuhub instance** (left up since the
D-034 benchmark session, still billing at $0.46/hr, reachable via `scripts/cloud/remote.env`) —
the user chose cloud over local Mac when asked, since it was already paid-for/idle and ~50x
faster (12.7-13.4 min wall-clock per run vs the Mac baseline's 2.4hr). **This is the first real
(non-benchmark-sweep) confirmation that a full training loop actually achieves the throughput
D-032/D-034's forward+backward-only sweeps predicted**: ~126K tok/s observed vs the sweep's
~630K peak (sweep numbers use a GPU-tuned micro_batch=64; these runs kept the Mac's
micro_batch=16 to hold the ablation config fixed, so the two numbers aren't directly comparable,
but 126K is well within plausible range and confirms the pipeline works end-to-end on real
gpuhub CUDA hardware, not just in isolated fwd/bwd sweeps).

**Options considered:** re-run the seed=1337 baseline a 4th time for symmetry (rejected — the
existing baseline run is byte-for-byte the same config modulo seed, re-running it would only
burn GPU time to confirm what's already registered); run all 3 seeds on Mac (rejected by the
user in favor of the idle already-billing cloud instance).
**Why this matters:** phase 5's entire verdict methodology (Wave A onward) depends on having a
noise floor to compare against — without it, a 0.01 val_loss difference between a technique and
its baseline would look meaningful when it's actually noise.
**Impacts:** `docs/EXPERIMENTS.md` (noise-floor section + rule), `experiments/registry.csv` (2
new rows), `experiments/20260712_p5_s-seed-{1338,1339}/` (config+metrics+samples+notes.md,
checkpoints left on the remote instance only — not pulled back, not needed for comparison).
**Revisit if:** a Wave A-G run shows a delta close to the 0.015-0.02 boundary — consider a 4th/
5th seed to tighten the estimate before trusting a borderline verdict either way.

## D-036 — Wave A (norms & activations) complete: QK-norm is a real win, SwiGLU confirmed over GELU, post-norm fails by stagnation not blow-up, RMSNorm-vs-LayerNorm is a wash (2026-07-12, phase 5, RW-3's live 5090 instance)
**Decision:** Ran all 4 Wave A ablations (`docs/phases/phase5_ablations.md`) same-session as the
seed-noise study, same RTX 5090 instance, same baseline (`20260711_p4_s-baseline`), same seed
(1337, since Wave A isn't testing seed variance) — only the model config axis under test differs
each time. **No new model code was needed**: `norm`/`norm_position`/`ffn`/`qk_norm` were all
already wired in phase 3 (`src/llmlab/model/{norms,block,ffn,attention}.py`); this wave was
config+run+analysis only, verified locally (`GPT(cfg)` builds + param-count sanity check for all
4 variants) before spending any GPU time.

**Results (val_loss delta vs baseline, judged against the D-035 noise floor of 0.0150):**
| Ablation | final val_loss | delta | verdict |
|---|---|---|---|
| RMSNorm -> LayerNorm | 3.4878 | -0.0158 | borderline (right at noise floor) |
| pre-norm -> post-norm | 6.8810 | +3.377 | negative result, as the spec predicted |
| SwiGLU -> GELU (param-matched, ffn_mult 8/3->4.0) | 3.6764 | +0.173 | real, robust — SwiGLU wins |
| +QK-norm | 3.4414 | -0.0622 | real, robust — **best of the wave** |

Every delta was checked against the *full per-checkpoint trajectory* (not just the final step)
per EXPERIMENTS.md's judging rule — RMSNorm/LayerNorm's gap stayed near the noise floor the whole
second half of training (genuinely marginal, not just a lucky final read); SwiGLU/QK-norm's gaps
were consistent or widening throughout (genuinely real, not early-training noise that faded).

**Post-norm's failure mode is worth recording precisely**: it did NOT diverge/spike
(grad_norm stayed <=1.52 throughout, under grad_clip=1.0) — it stagnated near loss~6.8 by step
~150 and never moved again, with degenerate punctuation-soup samples. This is a different
(and arguably more instructive) failure mode than "instability" usually implies — matches Xiong
et al. '20's mechanism (un-normalized residual stream dilutes/noises later-layer gradients) more
precisely than a generic "post-norm is unstable" framing would suggest.

**QK-norm's win was a genuine surprise**: went in expecting a small-or-negligible effect at
S-tier/15-layer depth (QK-norm is usually framed as a larger-scale stability aid), but it was
instead the single best result of the wave, with the gap *widening* (not fading) over training —
a real optimization-quality effect, not a lucky start. Recommending it as a new default going
forward is a genuine update to D-016's baseline recipe, not just a confirmation of an existing
choice.

**Options considered:** re-run RMSNorm/LayerNorm at more seeds to resolve the borderline result
now (rejected — not worth the GPU time/session length to resolve a comparison whose practical
answer, "keep RMSNorm for its lower compute cost," doesn't change either way); treat post-norm's
stagnation as a bug and debug/retune its warmup (rejected — the spec explicitly wants this as a
negative-result control at fixed compute, not a best-effort post-norm implementation).
**Why this matters:** first real use of the D-035 noise floor to separate "real" from "noise"
verdicts, and it already mattered in practice (RMSNorm/LayerNorm's result would have looked like
"LayerNorm wins" without it).
**Impacts:** `experiments/registry.csv` (4 new rows), `experiments/20260712_p5_s-wave-a-
{layernorm,postnorm,gelu,qknorm}/` (config+metrics+samples+notes.md; checkpoints left on the
remote instance, not pulled back), `docs/results/wave_a_norms_activations.png` (comparison
figure), `docs/results/ablation_log.md` (new file, 5-line-per-wave summary log per phase 5's
exit-criteria requirement). `docs/results/recipe.md` (phase 9 input) not yet written — waiting
until more waves land per the phase-5 exit criteria ("best-found recipe" implies waves A-D done).
**Revisit if:** a later wave's recipe interacts with QK-norm in an unexpected way (e.g. Wave C's
MLA has its own decoupled-RoPE-key mechanism that might make qk_norm redundant or conflicting —
check when implementing MLA).

## D-037 — Wave B (positional encodings) complete: ALiBi outperforms RoPE and extrapolates cleanly, sinusoidal is a genuine surprise-worst, RW-5's forward() fix lands (2026-07-12, phase 5)
**Decision:** Ran all 4 Wave B ablations (`docs/phases/phase5_ablations.md`) same session/instance
as Wave A, same baseline (`20260711_p4_s-baseline`, RoPE) and seed (1337). Before running,
implemented **RW-5's fix**: `GPT.forward()` (`src/llmlab/model/gpt.py`) previously hard-rejected
ANY sequence longer than `max_seq_len` for every `pos_encoding`; changed the guard to only apply
to `learned`/`sinusoidal` (whose position tables are physically sized to `max_seq_len`) — RoPE/
ALiBi/none compute position info on the fly per forward call and have no such limit. Updated
`tests/test_model.py`'s `test_exceeding_max_seq_len_raises` into two parametrized tests
(`..._raises_for_bounded_encodings` for learned/sinusoidal, `..._allowed_for_unbounded_encodings`
for rope/alibi/none) — a deliberate behavior change, not a silent one. Full suite (59 local/mps+
cpu, 42 remote/cuda-only) green after the change. Also wrote `scripts/eval_extrapolation.py` (new
permanent script, not a one-off): loads a run's checkpoint + config, builds a fresh
`MixedSourceLoader` at any seq_len against `val_sources`, reports val_loss/ppl or the expected
`ValueError` for bounded encodings — reusable for any future length-extrapolation work.

**Results (val_loss delta vs RoPE baseline at trained length, seq_len=512):**
| pos_encoding | final val_loss | delta | verdict |
|---|---|---|---|
| learned | 3.7311 | +0.227 | real, worse |
| sinusoidal | 4.9896 | +1.486 | real, WORST — surprise |
| **alibi** | 3.4830 | **-0.021** | real, best |
| none (NoPE) | 3.6997 | +0.196 | real, worse |

**Length-extrapolation probe (train@512, eval ppl@512/1024/2048), the headline result of this
wave:**
| pos_encoding | ppl@512 | ppl@1024 | ppl@2048 |
|---|---|---|---|
| rope (baseline) | 33.24 | 36.79 | 45.68 (degrades gracefully) |
| **alibi** | 32.56 | 32.08 | **31.67 (IMPROVES)** |
| none (NoPE) | 40.43 | 67.18 | **731.91 (collapses)** |
| learned/sinusoidal | 41.73/146.87 | ValueError (physically bounded) | ValueError |

**ALiBi beating RoPE, and improving rather than degrading under extrapolation, is a clean
small-scale reproduction of the ALiBi paper's headline claim** — genuinely useful evidence for
any future long-context decision (RW-5's other half: the phase-9 capstone's chat-context goal
wants ~2048+ tokens of usable context; this result argues ALiBi deserves serious consideration
there, not just RoPE by default). **Sinusoidal losing to `learned` by such a wide margin (+1.486
vs +0.227) was not expected** — going in, both were assumed roughly equivalent (same "additive
at the input, absolute position" mechanism, sinusoidal just without learnable parameters); the
gap suggests the model benefits substantially from being able to *adapt* its position
representation, even at this small scale, more than theory alone would predict.

**Options considered:** relax the `max_seq_len` guard for ALL encodings including learned/
sinusoidal (rejected — would silently produce garbage/index-errors for those, since their tables
are genuinely finite; the current per-encoding guard gives a clear, intentional `ValueError`
instead); skip the length-extrapolation probe until RW-5 is "fully" fixed with a broader
refactor (rejected — the narrow, well-scoped fix applied here is exactly what RW-5 asked for and
unblocks Wave B without overreaching into unrelated code).
**Why this matters:** the length-extrapolation result is one of this project's clearest paper
reproductions to date, and directly informs a phase-9 decision (capstone context length, RW-5).
**Impacts:** `src/llmlab/model/gpt.py` (forward() guard), `tests/test_model.py` (2 tests
replacing 1), `scripts/eval_extrapolation.py` (new script), `experiments/registry.csv` (4 new
rows), `experiments/20260712_p5_s-wave-b-{learned,sinusoidal,alibi,nope}/` (config+metrics+
samples+notes.md; checkpoints stayed on the remote instance), `docs/results/
wave_b_positional_encodings.png`, `docs/results/ablation_log.md`. RW-5's phase-9 half (capstone
`max_seq_len` decision) is unaffected — still open, but now with an ALiBi-vs-RoPE data point to
inform it when that decision comes up.
**Revisit if:** a later wave or the phase-9 capstone needs a firm real-context-length decision —
re-run this same probe at M/L tier before trusting these S-tier-only numbers at larger scale
(sinusoidal's surprise result especially should be re-checked, since D-016/D-021's hyperparameters
were tuned around RoPE, not sinusoidal, and might not be a fair comparison at a different lr/
schedule).

## D-038 — Wave C (attention variants) complete + MLA implemented: quality is flat across MHA/GQA/MQA/MLA, so cache size decides; MLA reproduces DeepSeek-V2's "small cache, kept quality" at 10M params (2026-07-13, phase 5)
**Decision:** Implemented **Multi-head Latent Attention** (`MLAAttention`, DeepSeek-V2 §2) plus a
full **incremental KV-cache decode path** for all four attention variants, then ran the Wave C
ablation (`docs/phases/phase5_ablations.md`) on the RTX 5090 gpuhub instance (same
`remote.env` host as Waves A/B — singapore-b:25864, re-provisioned from the "genesis" image).

**Two design decisions made before running (the user's calls, surfaced at session start):**
1. **MLA sizing = head-dim-preserving** (not DeepSeek-faithful proportions): per-head Q/K =
   `nope_head_dim 32 + rope_head_dim 32 = 64` (== the baseline `head_dim`), `v_head_dim 64`,
   `kv_lora_rank 128`, `q_lora_rank 192`. Keeps head geometry identical to MHA for the fairest
   quality comparison; cache/token = `(128+32)·2B = 320 B/tok/layer`. Extended `MLAConfig` with
   `nope_head_dim`/`v_head_dim` (were missing).
2. **Full incremental KV-cache decode path** (not analytical-only): new `src/llmlab/model/
   kv_cache.py` (`KVCache` for MHA/GQA/MQA, `MLACache` for the latent+shared-rope-key), `cache=`
   threaded through `Attention`/`MLAAttention`/`Block`/`GPT.forward`, and `GPT.generate()`
   rewritten to prefill-once-then-1-token/step. Reusable for phase 8/9 chat. Cached decode is
   verified bit-exact vs full-sequence forward on cpu/mps/**cuda** (maxdiff ~1e-7,
   `test_cached_decode_matches_full_forward`).

**Forced implementation constraint (not a preference):** GQA "2 groups" is undefined at the
baseline's `n_heads=3` (3 ∤ 2), so the **entire wave runs at `n_heads=4, head_dim=64`** and the
**4-head MHA run (`20260713_p5_s-wave-c-mha`) is the wave's internal control**, NOT the 3-head
`p4_s_baseline`. All four variants share the 4-head geometry, so each gqa2/mqa/mla comparison is
single-variable.

**Results (val_loss @ 98.3M tokens, seed 1337, same harness as A/B; noise floor 0.0150 D-035):**
| variant | val_loss | Δ vs MHA control | cache B/tok/layer | vs MHA |
|---|---|---|---|---|
| GQA-2 | 3.5107 | −0.0205 | 512 | 2.0× smaller |
| MLA | 3.5146 | −0.0166 | 320 | 3.2× smaller |
| MHA (control) | 3.5312 | — | 1024 | — |
| MQA | 3.5498 | +0.0186 | 256 | 4.0× smaller |

**Reading:** quality is **nearly flat** (spread 0.039 ≈ 2.6× the noise floor) — at S-tier/98M
tokens the attention *type* barely moves loss, so the decision is made on **cache**. GQA-2 and
MLA both marginally *beat* MHA (each just past the noise floor) while cutting cache 2–3.2×; MQA is
the only real quality *loss* (+0.0186) but has the smallest cache. **MLA is the Pareto-interesting
point:** it dominates MHA (smaller cache, equal-or-better quality) and matches GQA's quality at a
smaller cache — a clean small-scale reproduction of DeepSeek-V2's central claim.

**KV-cache bytes measured analytically AND empirically** (`scripts/bench_inference.py` on the
5090, bf16) — empirical == analytical exactly (`docs/results/wave_c_inference_bench.csv`). **The
honest tok/s finding:** at 10M params single-stream decode is launch-overhead-bound, so the naive
`torch.cat` cache does NOT speed up latency vs full recompute, and MLA decodes ~25% slower/token
than MHA (extra down/up projections; we skipped the "weight absorption" trick and pre-allocated
cache — both noted in `notebooks/06_mla_explained.ipynb` §4). MLA buys cache *memory* with
*compute*; at this scale the payoff is memory (→ larger batch/longer context), not latency.

**Options considered:** (a) DeepSeek-faithful MLA proportions (nope 64 > v 64 > rope 32) —
rejected in favor of head-dim-preserving for a fair quality comparison; (b) analytical-only cache
measurement — rejected, the incremental decode path is needed for a real bytes/tok-s measurement
AND is reused in phase 8/9; (c) keep `n_heads=3` and drop GQA — rejected, GQA is a core Wave C
technique, so the 4-head wave with its own MHA control is the correct single-variable design.

**Why this matters:** Wave C is DeepSeek flagship #1 and the hardest implementation of the
project; MLA + the KV-cache path are now real, tested code the capstone can use. The verdict feeds
phase 9's recipe: MHA's full cache buys nothing at this scale.

**Verdict for phase 9's recipe:** default to **GQA** (2× cache cut, zero quality cost, trivial
code); reach for **MLA** when KV-cache memory is the binding constraint (long context / large
batch), accepting the decode-compute overhead that absorption + a pre-allocated cache would
remove; **MQA** only if cache is the single overriding constraint and a small quality hit is OK.

**Impacts:** `src/llmlab/model/config.py` (`MLAConfig` +nope/v dims), `attention.py`
(`MLAAttention`, `make_cache`, cached `Attention`), `kv_cache.py` (new), `block.py`/`gpt.py`
(cache threading, rewritten `generate`), `positional.py` (`RotaryEmbedding` offset),
`tests/test_model.py` (MLA + KV-cache tests, 82 pass), `scripts/bench_inference.py` +
`scripts/plot_wave_c.py` (new), `configs/model_s_attn_{mha,gqa2,mqa,mla}.yaml` +
`configs/train_s_wave_c_{...}.yaml` (new), `notebooks/06_mla_explained.ipynb` (new),
`experiments/20260713_p5_s-wave-c-{mha,gqa2,mqa,mla}/` (4 runs, config+metrics+samples+notes;
checkpoints stayed on the remote), `experiments/registry.csv` (4 rows),
`docs/results/wave_c_attention_variants.png`, `docs/results/wave_c_inference_bench.csv`,
`docs/results/ablation_log.md`.

**Revisit if:** the phase-9 capstone needs the KV-cache decision at M/L tier — re-run the bench
there (bytes/token scales with n_layers·d, and the MLA-vs-GQA quality gap may open or close at
larger scale/longer training). If MLA is chosen for the capstone, implement weight absorption +
pre-allocated cache first (this session's decode path is correct but not throughput-optimized).

## D-039 — Wave D (optimizers & schedules) complete: Muon is the single biggest lever found so far, WSD/late-decay beats cosine, Lion/z-loss/AdamW-hparam runs are honest nulls or need re-tuning (2026-07-13, phase 5)
**Decision:** Implemented **Muon** (Jordan '24 Newton-Schulz-orthogonalized momentum, hybrid with
AdamW for embeddings/norms per the nanoGPT speedrun recipe) and **Lion** (Chen '23 sign-based
update) as new `torch.optim.Optimizer` subclasses (`src/llmlab/train/optimizers.py`), generalized
`Trainer`'s single-AdamW assumption into a list-of-optimizers design (`_build_optimizers`,
`OptimConfig.optimizer`), generalized the lr schedule from hardcoded cosine into a dispatched
`_schedule_multiplier` (`cosine`/`wsd`/`constant`, `OptimConfig.schedule`), and added PaLM '22
z-loss (`OptimConfig.z_loss_weight`, computed in `train_step` from the logits `GPT.forward()`
already returns — no model-code change needed). Ran 13 short S-tier runs on the RTX 5090 gpuhub
instance (same host as Waves A-C, singapore-b:25864, ~42 min wall-clock total for the first 11).

**Design decisions made before running:**
1. **New Wave D control, not a reuse of `p4_s_baseline`.** Switched `micro_batch`/`grad_accum`
   from the Mac-tuned 16/8 to the RTX 5090's measured S-tier sweet spot 64/2 (D-030) — same
   65,536 tok/step effective batch, but the loader's stateless `(seed, step)` sampling is keyed
   off `step * grad_accum + micro`, so this changes which data offsets land on which step even at
   an identical effective batch. Same reasoning as Wave C's n_heads=4 control (D-038). Confirmed
   within noise of `p4_s_baseline` (3.4977 vs 3.5037).
2. **Muon hybrid split:** 2D hidden weight matrices (attn/ffn projections) -> Muon
   (`muon_lr=0.02`, momentum=0.95, 5 Newton-Schulz steps); embeddings/norms -> a separate,
   no-decay AdamW (`lr=1e-3`). Both schedule off the same warmup/decay *shape*
   (`lr_at_step(step, cfg, base_lr=...)` now takes an optional peak-lr override) at their own
   peak values — lets one `_schedule_multiplier` serve both optimizers.
3. **Lion's hyperparameters were NOT swept** — used the paper's recommended one-shot conversion
   from the AdamW recipe (lr /3.3 -> 3e-4, wd x3 -> 0.3) rather than spending session time tuning
   it, given the wave's run budget. This matters for how the result should be read (see below).
4. **Batch-size study held lr fixed** (deliberately not applying the linear-scaling rule) to
   demonstrate the batch/steps/lr coupling directly, at a fixed ~98.3M-token budget across all
   three effective-batch points (0.06M control / 0.25M / 1M tok/step).
5. **WSD multi-budget bonus implemented as a real fork**, not simulated: `wave_d_constant`
   (warmup+flat-forever, no decay) ran to completion first; its real step-1500 checkpoint was
   then reused (copied into two new run folders on the remote) for two independent decay-tail
   continuations (`--resume`) at different total budgets (+10%/+26.7% tokens), with
   `wsd_decay_ratio` set per-fork so decay starts exactly at the resume point.

**Results (val_loss vs control 3.4977, judged against the D-035 noise floor of 0.015-0.02):**
| Run | val_loss | delta | verdict |
|---|---|---|---|
| Muon | 3.3432 | **-0.1545** | REAL, ROBUST, best of the wave |
| Lion | 3.9203 | +0.4226 | REAL as run, but un-tuned — not a fair verdict on Lion |
| WSD | 3.3764 | -0.1213 | REAL win |
| constant (no decay) | 3.4303 | -0.0674 | REAL win (surprising — beats cosine) |
| z-loss (1e-4) | 3.5029 | +0.0052 | null (within noise) |
| grad_clip off (1e6) | 3.5192 | +0.0215 | REAL but undramatic — no spike |
| batch 0.25M tok/step | 4.2567 | +0.759 | REAL, as predicted |
| batch 1M tok/step | 5.3942 | +1.8965 | REAL direction, magnitude confounded (see below) |
| AdamW wd=0 | 3.4935 | -0.0042 | null |
| AdamW beta2=0.999 | 3.5099 | +0.0122 | null |
| WSD fork, short (+10% tok) | 3.3220 | -0.1083 vs fork point | REAL |
| WSD fork, long (+26.7% tok) | 3.2768 | -0.1535 vs fork point | REAL, best number in the wave |

**The schedule hierarchy is the wave's cleanest finding:** WSD (-0.1213) > constant (-0.0674) >
cosine (control) — decaying the LR only at the very end beats never decaying, which in turn
beats cosine's continuous decay from step 30 onward. WSD was already slightly ahead of cosine
by step 500-1000, BEFORE its own decay phase even starts (decay begins at step 1200) — evidence
that cosine's early, continuous decay costs real ground well before its own endpoint.

**Muon's gap narrows but never closes** (-0.267 @ step500 -> -0.185 @ step1000 -> -0.155 final) —
matches the nanoGPT speedrun's framing of Muon as primarily a *convergence-speed* accelerator
(biggest edge early) rather than a higher asymptotic ceiling.

**grad-clip-off did NOT produce the spec's predicted "spike."** `clip_grad_norm_` always returns
the PRE-clip norm regardless of whether clipping is subsequently applied, so the logged
`grad_norm` metric is identical (max 5.51 at step 0, both runs) whether or not clipping happens —
the real effect (control's train_loss consistently ~0.02-0.1 lower at every early checkpoint) is
steady and small, not a dramatic single event. At this depth (15 layers, pre-norm) with a 30-step
warmup, the architecture is already stable enough that grad_clip=1.0 rarely binds hard.

**Two honest confounds flagged rather than papered over** (matches this project's established
self-correction culture, e.g. D-032->D-034): (a) Lion's result reflects one un-tuned
hyperparameter guess, not a real Lion-vs-AdamW/Muon verdict; (b) the 1M-tok/step batch run's
`warmup_steps=30` wasn't scaled down, so 32% of its 94-step budget is warmup — the direction
(bigger batch without lr scaling undertrains at fixed tokens) is confirmed by the cleaner 0.25M
point, but the 1M point's magnitude is inflated by this oversight.

**Options considered:** fusing Muon+AdamW into one combined optimizer class (rejected — two
plain `torch.optim.Optimizer` instances the `Trainer` steps/checkpoints as a list is simpler and
keeps each optimizer's own logic/tests independent); tuning Lion's lr before reporting (rejected
for this session's time budget — flagged as a follow-up instead of blocking the wave); simulating
the WSD multi-budget bonus via config math alone instead of a real checkpoint fork (rejected —
the whole point is demonstrating it works on real trained weights, and the extra GPU cost was
trivial, ~1 min).

**Why this matters:** Muon is the single largest effect-size finding in the project to date
(>10x the noise floor), and the WSD-vs-cosine schedule hierarchy + multi-budget fork are
directly actionable for phase 9's recipe. This completes the M2 milestone (Waves A-D all done).

**Impacts:** `src/llmlab/train/optimizers.py` (new: `Lion`, `Muon`, `zeropower_via_newtonschulz5`),
`src/llmlab/train/config.py` (`OptimConfig` +optimizer/muon_*/schedule/wsd_decay_ratio/
z_loss_weight fields), `src/llmlab/train/trainer.py` (`_build_optimizers`,
`_schedule_multiplier`, `_split_params_by_ndim`, list-of-optimizers checkpointing, z-loss in
`train_step`), `tests/test_optimizers.py` (new, 8 tests), `tests/test_trainer.py` (+7 tests:
schedule shapes, base_lr override, Lion/Muon resume round-trips, z-loss sanity) — full suite 96
passed locally (cpu/mps) + 66 passed remotely (cuda). `configs/train_s_wave_d_*.yaml` (13 new),
`experiments/20260713_p5_s-wave-d-*/` (13 runs, config+metrics+samples+notes.md; checkpoints
stayed on the remote except where reused for the WSD fork), `experiments/registry.csv` (13 rows),
`docs/results/wave_d_optimizers_schedules.png`, `scripts/plot_wave_d.py`,
`docs/results/ablation_log.md`.

**Revisit if:** a phase-9 recipe decision needs Lion re-tuned properly (sweep lr/wd before using
its result either way); the M/L tier wants to confirm Muon's speedup holds at larger
model/context size (this was S-tier/10M-params only); a future wave wants the grad-clip-off
"dramatic spike" demo specifically — would need a less-stable setup (higher lr, no warmup, or a
much longer run) to actually produce one at this architecture's depth.

## D-040 — Wave E results: bf16/torch.compile are free speed wins, gradient checkpointing trades ~27% speed for ~1.72x memory, batch factorization is loss-invariant but not wall-clock-invariant, untied embeddings win but aren't param-matched  (2026-07-13, phase 5)
**Decision:** Ran phase 5's Wave E (efficiency & memory: bf16 vs fp32, gradient checkpointing,
micro-batch/accum equivalence, weight tying, torch.compile, activation-memory-vs-seq_len) — 6
S-tier training runs + a standalone memory-sweep benchmark, all on the RTX 5090
(gpuhub singapore-b:25864). Unlike Waves A-D, four of five training-run axes are NULL results on
loss BY DESIGN (they're efficiency knobs that shouldn't change what's computed) — the real
findings are speed and memory numbers, verified as real effects rather than assumed.

**New code required (this wave needed genuine new trainer/model plumbing, unlike Waves A-C which
were config+run+analysis only):** `TrainConfig` gained `precision` (bf16/fp32),
`gradient_checkpointing`, `compile` fields. `GPT` gained a runtime `gradient_checkpointing`
attribute (deliberately NOT a `ModelConfig` field — it's a compute/memory trade-off that doesn't
change what's computed, not an architecture choice) wrapping each block in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when `self.training` and no KV
cache is active. `Trainer` gained `_autocast()` (dispatches bf16 autocast vs a plain
`nullcontext` for fp32 — eval deliberately stays ungated by this, it always ran in fp32 even
before this wave, so every wave's val_loss stays measured the same way regardless of what
precision a given run trained under) and a `torch.compile` attempt at init time, guarded by
try/except so a failure is logged and degrades to uncompiled rather than crashing.
**Checkpointing correctness fix made in the same pass:** `save_checkpoint`/`load_checkpoint`/
`num_params()` now go through a new `self._raw_model` reference (the pre-compile module) instead
of `self.model` directly — `torch.compile`'s wrapper's `state_dict()` key-naming behavior is
version-dependent, so routing checkpoints through the always-uncompiled reference removes that
risk entirely rather than trusting current PyTorch's behavior to hold.

**Results (control: `20260713_p5_s-wave-d-control`, reused from Wave D; full writeup
`docs/results/ablation_log.md`, figure `docs/results/wave_e_efficiency_memory.png`):**
- **bf16 vs fp32:** NULL on quality (+0.0083, noise), REAL on speed — fp32 is ~35% slower
  (~296.8K vs ~455.1K tok/s). Confirms D-009's bf16-by-default choice was correctly free.
- **Gradient checkpointing:** NULL on quality (-0.0088, noise, as expected for an exact
  recompute), ~27% slower at this size with no memory upside (512/mb64 already fits). The real
  payoff is a separate `bench_activation_memory.py` seq_len sweep (new script): a consistent
  **~1.72x peak-memory reduction at every seq_len tested (128-1024)**, and it buys exactly one
  more doubling of context before OOM on the 5090's 32GB (2048 fits checkpointed, OOMs
  uncheckpointed; checkpointed itself OOMs at 4096).
- **Micro-batch/grad-accum equivalence:** three factorizations of the same 128-seq effective
  batch (control mb=64/accum=2, mb=32/accum=4, mb=128/accum=1) all land within noise on loss —
  confirms grad accumulation is mathematically exact, not an approximation. But wall-clock varies
  more than 2x across them (mb=32/accum=4 slowest at ~248.2K tok/s, mb=128/accum=1 fastest at
  ~525.1K tok/s) for IDENTICAL FLOPs — confirms D-022's launch-overhead-bound finding and gives a
  concrete rule: **always prefer the largest micro-batch that fits.**
- **Weight tying off:** REAL but caveated — untied wins (-0.0278, just past noise) but this is
  **not a param-matched comparison** (12.79M vs control's 9.71M tied, +31.6% params) — the win
  may just be extra capacity, not evidence tying costs quality at a fixed layer shape. Does not
  overturn D-016 (which was a cost-efficiency argument, not a quality one); a param-matched
  rerun is flagged as a future follow-up, not done this wave (time-budget call).
- **torch.compile:** NULL on quality (+0.0014, noise), REAL win on speed — **fastest run in the
  wave** (~535.4K tok/s, ~18% faster than uncompiled), compiled cleanly on CUDA with zero
  fallback/graph-break issues at this size. CLAUDE.md's MPS-unreliable caveat is untouched (ran
  on the 5090, not tested on Mac this wave).

**Why:** the phase-5 spec explicitly scoped Wave E as "measurement-heavy" — the goal is knowing
which efficiency knobs are free (bf16, compile), which have honest costs with a specific payoff
(gradient checkpointing), and which are non-issues that just needed confirming (batch
factorization's loss-invariance). Verifying each empirically rather than assuming from
first-principles reasoning caught one real, actionable number in each case (e.g. the exact
~1.72x memory ratio, the exact 2x+ wall-clock spread across factorizations) that a plausible
prior guess would have gotten only roughly right.

**Impacts:** `src/llmlab/train/config.py` (+3 fields), `src/llmlab/train/trainer.py`
(`_autocast`, `_raw_model`, `compile_status`, checkpoint routing fix), `src/llmlab/model/gpt.py`
(`gradient_checkpointing` attribute + block-wrap), `tests/test_model.py` (+2 tests: checkpointing
loss/grad parity, eval-mode never checkpoints), `tests/test_trainer.py` (+4 tests: flag reaches
the model, fp32 disables autocast, unknown precision raises, compile-disabled-by-default leaves
`model is _raw_model`) — 89 passed locally (cpu/mps), 64 passed remotely (cuda-only device
matrix, pre-existing gap: `tests/test_model.py`'s `DEVICES` fixture only ever adds mps, never
cuda — not fixed this wave, out of scope). `scripts/bench_activation_memory.py` (new),
`scripts/plot_wave_e.py` (new), `configs/train_s_wave_e_*.yaml` (6 new) + `configs/
model_s_notie.yaml` (new), `experiments/20260713_p5_s-wave-e-*/` (6 runs, config+metrics+
samples+notes.md, checkpoints stayed remote), `experiments/registry.csv` (+6 rows, real
verdicts not placeholders), `docs/results/wave_e_efficiency_memory.png`,
`docs/results/wave_e_activation_memory{,_gradckpt}.csv`, `docs/results/ablation_log.md`.
Completes Wave E of the Phase 5 checklist (Waves A-D already done as M2); Waves F-G remain.

**Also found and fixed during sync setup (not a decision, but worth a paper trail):** a
trailing-slash rsync bug (`rsync ... src/ configs/ ... dest/` copies `src/`'s CONTENTS into
`dest/` rather than creating a `dest/src/` subdirectory) briefly created a stray, incomplete
top-level `llmlab/` package on the remote pod (missing the `data` subpackage) that shadowed the
real `src/llmlab/` via Python's cwd-first `sys.path` resolution, breaking `tests/test_trainer.py`
collection. Removed the stray directory and redid the sync without trailing slashes on the
source args. No project code was affected — purely a one-time remote-filesystem cleanup.

**Revisit if:** a phase-9 M/L-tier or the capstone run wants to stack bf16+compile+the
largest-micro-batch-that-fits together (not measured jointly this wave, each was isolated
against the same control) — worth a quick joint-speedup confirmation before relying on the
product of the two independent percentages; or if the weight-tying question needs settling
properly (param-matched untied run) before phase 9's recipe finalizes.

## D-041 — Checkpoint archival to R2: strip optimizer state by default, keep full only at named fork points; server→R2 direct, no Mac round-trip  (2026-07-16, phase 5)
**Decision:** Built `scripts/cloud/archive_checkpoints.py` (runs on the pod, needs torch) +
`scripts/cloud/push_checkpoints.sh` (Mac-side wrapper: scp the archiver over, run it, `rclone
copy` the staged output straight from the pod to `r2:${R2_BUCKET}/experiments/`, verify, clean
up the staging dir — never touches `experiments/` itself). Policy:
- Every run archives `config.yaml` + `metrics.jsonl` + `notes.md` + `samples/` as-is (KB-scale)
  plus `ckpt/best.pt` with `optimizer_state_dict(s)` stripped out — model weights only
  (~39MB/run at S-tier vs ~111MB full). Ablation runs are reproducible from config+seed (the
  project's own standard), so an archived copy doesn't need to be resumable, just reloadable
  for eval/generation/inspection.
- Runs that have had a real `--resume` fork off them are named explicitly (currently just
  `20260713_p5_s-wave-d-constant`, the WSD multi-budget bonus's shared checkpoint, D-039) and
  get BOTH `ckpt/best.pt` and `ckpt/latest.pt` archived in full (optimizer state intact) —
  these need to stay actually resumable.
- Pushed **server→R2 directly** (rclone already installed+credentialed on the pod per D-026),
  not via the Mac — avoids a home-bandwidth round-trip for data that's disposable at the Mac end
  anyway; R2 has zero egress either direction so cost is identical.
- Nothing is deleted from the pod's data disk by this script (user's explicit call this
  session, since the 50GB data disk had 37GB free — not under real pressure yet). R2 becomes an
  additional durable copy, not (yet) a replacement for the pod copy.

**First run (2026-07-16):** archived all 48 existing runs, 1.395 GiB (348 objects) pushed in
~110s. R2 bucket `llm` total is now 4.274 GiB (2.879 GiB tokenized data from D-026/RW-1 +
1.395 GiB experiments) — comfortably inside the free 10GB tier, nowhere near the user-approved
50GB ceiling even before accounting for future M/L-tier runs.

**Options considered:** keep last-N checkpoints per run (rejected — CLAUDE.md already settled
"latest.pt + best.pt only, no milestone hoarding," and the trainer already tracks `best.pt`
separately from `latest.pt` by val_loss, so "the last checkpoint isn't the best one" is already
handled in code, not a gap R2 needed to solve); archive full checkpoints for every run (rejected
— 3x the size for zero benefit on runs nobody will ever resume; R2 free-tier math would still
work but wastes the margin needed for M/L-tier runs later); route through the Mac first
(rejected — pure overhead, R2 has no egress cost either direction so server→R2 direct is
strictly better).

**Also found while investigating this (not this decision, but logged for the paper trail):**
the pod's git working tree is stale (`HEAD` at Wave D's `3d330cc`, several commits behind the
Mac's `86edd50`) with uncommitted drift in `PROGRESS.md`/`DECISIONS.md`/`registry.csv`/several
`src/llmlab/model/*.py` files — caused by the trainer writing directly to the pod's
`registry.csv` (never committed there) while the Mac's copy was separately polished and
committed. Confirmed this is **not** a checkpoint-safety issue (`experiments/**/ckpt/` and
`wandb/` are gitignored and untracked on both machines — `git pull` cannot touch them under any
circumstance) but a plain `git pull` on the pod would likely refuse to run until that drift is
discarded (`git checkout -- .` on the pod, Mac's versions are authoritative). Not fixed this
session — flagged for whoever next needs to `git pull` on this pod.

**Revisit if:** the pod's data disk actually fills up (currently 37GB free of 50GB) — switch
the "keep on pod too" default to "delete after verified push"; or if a future run needs to be a
fork point after the fact — add its run_id to `push_checkpoints.sh`'s `FORK_POINTS` arg and
re-run (rclone only pushes the newly-full checkpoint, cheap).

## D-042 — wandb turned on: account credentials in `.env`, all 33 offline runs synced, `--wandb-online` added for live cloud monitoring  (2026-07-16, phase 5)
**Decision:** User created a wandb account and provided `WANDB_API_KEY` this session. Stored in
root `.env` (gitignored, same file as R2 credentials, D-026) + placeholder added to
`.env.example`. New doc `docs/WANDB.md` covers the full setup for future sessions. Three parts:

1. **Entity correction (verified via API, not assumed):** the user-provided
   `WANDB_ENTITY=adityaram0001` is invalid — `adityaram0001` is the account's *username*, not an
   entity slug. Queried `wandb.Api().viewer`/`api.default_entity` directly and found the real
   entity is `adityaram0001-bbiq-technologies-private-limited` (an org/business entity the
   account is scoped under — the account has no separate personal entity). Using the wrong value
   silently broke every sync: the CLI printed `Syncing: ... done.` for all 33 runs on the first
   attempt, but the pod's debug log showed every single one had actually failed server-side
   (`CommError: entity adityaram0001 not found during upsertBucket`) — **the CLI's "done." only
   means the local process exited, not that the upload succeeded.** Caught by checking
   `api.runs(...)` afterward instead of trusting the sync command's own stdout — a second real
   instance (after D-023, D-022's list-aliasing bug, D-032's incomplete sweep) of this project's
   working pattern: verify a claimed-successful operation against real state, not the tool's own
   "done" message. `.env` corrected + comment added explaining why, re-ran, then re-verified via
   `api.runs('adityaram0001-bbiq-technologies-private-limited/llm-lab')` — **33/33 runs present,
   all `state=='finished'`, spot-checked values matching known results** (e.g.
   `wave-a-postnorm`'s synced val_loss ≈6.88 matches D-036's stagnation finding).
2. **Historical sync**: `scripts/cloud/archive_checkpoints.py`'s sibling script
   `scripts/cloud/wandb_sync.sh` — scp's the Mac's `.env` to the pod, then runs `wandb sync` on
   every `experiments/*/wandb/offline-run-*` directory found there (all offline runs have always
   lived nested inside each run's own folder, per `trainer.py`'s `dir=str(run_dir)`, NOT a
   top-level `wandb/` — `sync_down.sh`'s dedicated top-level-`wandb/`-pull step has been silently
   dead code all along, harmless since the main `rsync experiments/` already covers the real
   path). Project name stays `"llm-lab"` (D-005's original choice, `TrainConfig.wandb_project`
   default) — deliberately NOT overridden by any env var, so it can't drift per-machine.
3. **Live monitoring going forward**: added `--wandb-online` to `scripts/train.py` (mirrors the
   existing `--device` override pattern) — overrides one run's `wandb_mode` to `"online"` at
   launch without touching D-009's offline-by-default for every other run. Requires
   `WANDB_API_KEY` present wherever `train.py` runs; `wandb_sync.sh` pushing `.env` to the pod
   covers that for cloud runs.

**Options considered:** making online the global default once a key exists (rejected — the same
`.env` now lives on the Mac too, so this would silently put every local smoke-test/dev run
online, reversing D-009's deliberate offline-by-default without being asked); a new
`WANDB_PROJECT` env var (rejected — code always passes `project=` explicitly to `wandb.init()`,
so an env var would either be ignored or, worse, invite a future accidental override of D-005's
established project name).

**Why this matters beyond the fix itself:** the wrong entity would have kept silently "succeeding"
indefinitely — nothing in the CLI's own output would ever have surfaced the failure. Any future
wandb-related script in this project should verify against `wandb.Api()` after a sync/init, not
just check the process exit code or stdout.

**Revisit if:** the account ever gains a real personal entity (unlikely for an org-scoped
account) — re-check `api.default_entity` before assuming `WANDB_ENTITY` is still correct.

## D-043 — Confirmed: 16 of Waves A/B/C's cloud runs used the Mac-tuned micro_batch=16, not the 5090's mb=64 sweet spot; added a runtime warning since the doc alone got missed  (2026-07-16, phase 5)
**Decision:** While setting up wandb (D-042), grepped every `configs/train_s_wave_*.yaml`'s
`micro_batch` value and found the exact scenario `docs/CLOUD_GPUHUB.md` §10 already warned
about (written 2026-07-12, before Wave A even ran) had actually happened, silently, for two full
waves:
- **Wave A (4 runs), Wave B (4 runs), Wave C (4 runs, MHA/GQA2/MQA/MLA) — all 12 used
  `micro_batch=16`** (the Mac/MPS-tuned plateau value, D-022), not the RTX 5090's measured S-tier
  sweet spot of `micro_batch=64` (~629,837 tok/s vs whatever mb=16 achieves — the sweep in
  CLOUD_GPUHUB.md §10 shows throughput scaling hard with micro-batch on CUDA, unlike Mac's flat
  D-022 curve, so this is a real, not cosmetic, gap).
- **Wave D onward (starting `wave_d_control`) already self-corrected to `micro_batch=64`** —
  nobody flagged this explicitly at the time, it just happened to get fixed. Wave E's own
  micro-batch/accum-factorization axis (mb=32/64/128) is the one legitimate exception where a
  non-64 value is the deliberate independent variable, not a mistake.
- **None of Waves A-C's quality verdicts are affected** — val_loss is compute-identical
  regardless of micro-batch/accum factorization (this is exactly what Wave E's own ablation
  independently confirmed, D-040). The only cost was wall-clock/GPU-hours: those 12 runs likely
  ran at a small fraction of the 5090's achievable throughput for no benefit, i.e. paid-for GPU
  time was left on the table, not a scientific error.

**Fix:** added a runtime warning in `Trainer.__init__` (`src/llmlab/train/trainer.py`) — prints
loudly whenever `device.type == "cuda"` and `cfg.batch.micro_batch <= 16`, pointing at
CLOUD_GPUHUB.md §10. Not a hard error (Wave E's deliberate low-micro-batch runs would trip a
naive check, though none of them are actually ≤16 in practice — the threshold was chosen to
match the one real known-bad value, not to fire on every small micro-batch). Also strengthened
CLOUD_GPUHUB.md §10 with an explicit "this already happened" callout above the sweet-spot table.

**Why a doc wasn't enough:** the sweet-spot table and its warning sentence were already written
in CLOUD_GPUHUB.md *before* Wave A ran (2026-07-12) — reading it wasn't the failure mode, reusing
an earlier wave's YAML as a copy-paste starting point (which still had the Mac default) was. A
runtime nag fires regardless of which doc did or didn't get re-read, which is a stronger
guarantee for a project doing many more waves ahead (F, G, plus M/L-tier confirmation runs).

**Options considered:** retroactively re-running Waves A-C at mb=64 (rejected — their val_loss
verdicts are already correct and equal-tokens/equal-wallclock comparisons within each wave are
still internally consistent since every run in a given wave used the same micro_batch; re-running
would cost real GPU-hours to fix a number that was never wrong, only slower than it needed to be);
doc-only fix (rejected — already tried, already failed once); hard error instead of a warning
(rejected — would break Wave E's legitimate micro-batch ablation and any future one like it).

**Revisit if:** a future wave's config generation becomes scripted/templated rather than
hand-copied YAML — the sweet-spot value could then be injected automatically per-device instead
of relying on a human noticing the warning.


## D-044 — Wave F (DeepSeek specials): implemented MoE + MTP; DeepSeekMoE reproduces its headline win; caught and fixed a real val_loss measurement bug before drawing any conclusion  (2026-07-16, phase 5)
**Decision:** Implemented the phase 5 spec's Wave F in full: `src/llmlab/model/moe.py`
(`MoEFFN` — 8 fine-grained routed experts + 1 shared, top-2 routing, expert hidden dim sized so
ACTIVE params/token match the dense baseline's FFN, per DeepSeekMoE section 3.1's fine-grained
segmentation; two balancing methods, `aux_loss` Switch/GShard-style and `bias_free` DeepSeek-V3
S2.1.2's per-expert selection-only bias) and `src/llmlab/model/mtp.py` (`MTPHead` — sequential
Multi-Token-Prediction depths, each combining the previous depth's hidden state with the true
teacher-forced next-token embedding through one more transformer `Block`, sharing the main
model's `final_norm`+`lm_head`). Both wired through `Block`/`GPT`/`Trainer` with no new
NotImplementedError guards — `moe`/`mtp` config fields are now fully live. +34 tests (127 local
cpu/mps, 98 remote-cuda, all pass).

**Design choices worth recording:**
- **Expert sizing:** `expert_hidden = round(dense_hidden / (n_shared + top_k))` — at S-tier
  (d_model=192, dense hidden=512, 8 routed + 1 shared, top-2) this gives expert_hidden=171,
  landing total active FFN params within ~0.1% of the dense control's 4.42M while total FFN
  capacity grows to 13.32M (9 expert-equivalents vs 1) — total model 18.61M vs control's 9.71M.
- **Routing:** router is a plain linear + softmax over all `n_experts`; combination weights for
  the selected top-k always come from this UNBIASED softmax (`gate_probs`), renormalized to sum
  to 1 across the selected experts. `bias_free`'s per-expert bias is added ONLY to the logits
  used for top-k SELECTION, never to the combination weight and never part of any loss —
  matches DeepSeek-V3's explicit design (bias moves who gets picked, not how much a picked
  expert's output counts).
- **aux_loss formula:** per-layer `n_experts * sum(f_i * P_i)` (`f_i` = stop-grad routed-token
  fraction, `P_i` = mean softmax probability mass, both across the current batch), summed
  (not averaged) across all `n_layers` MoE layers before applying `aux_loss_weight` — this
  matters for interpreting the raw `moe_aux_loss` metric (~1.0/layer at good balance, so ~15 at
  15 layers), see the bug below.
- **bias_free update rule:** `routing_bias += update_rate * sign(mean_load - load)`, called once
  per OPTIMIZER step (not per micro-batch) by a new `Trainer` hook (`GPT.update_moe_bias`),
  aggregating load across all of that step's grad-accum micro-batches — gradient-free, a plain
  buffer update, no interaction with `loss.backward()`/`opt.step()`.
- **MTP simplification (flagged, not a bug):** the MTP block is always DENSE (non-MoE) even when
  the main trunk uses MoE FFN layers, and always mirrors the main trunk's attention type —
  avoids nesting a second independent MoE router+expert set behind one extra head. MTP requires
  `pos_encoding` in `{rope, alibi, none}` (raises `NotImplementedError` for learned/sinusoidal —
  those add position at the input embedding stage, which MTP heads never pass through; rope/
  alibi both inject position per-block, which the MTP block's fresh `Block` call reuses
  unmodified since the retained subsequence is always the ORIGINAL left-aligned prefix).

**A real bug found and fixed mid-wave — the more important story:** the first attempt at both
MoE runs computed `Trainer.evaluate()`'s `val_loss` from `GPT.forward()`'s COMBINED training
objective (main cross-entropy + `aux_loss_weight * moe_aux_loss`), not pure CE. Because
`moe_aux_loss` sums across all 15 layers (~15 at good balance) and `aux_loss_weight=0.01`, this
silently added ~+0.15 to the `aux_loss` run's reported metric while `bias_free`'s (correctly
exactly zero by design, since that method has no loss term at all) was unaffected by the same
bug — producing a fake ~0.15 "aux_loss balancing is worse" gap between the two runs that would
have read as a real, interesting finding if not caught. Caught by checking the raw numbers
against the D-035 noise floor before writing any verdict (0.15 is 7-10x the floor — implausibly
large for what should be a close comparison) — another instance of this project's now-recurring
pattern (D-022, D-023, D-032, D-042, D-043) of a plausible-looking automated result turning out
to be a measurement artifact, not a real effect. **Fix:** `GPT.forward()` now stores pure CE in
`self.last_aux_metrics["ce_loss"]` before any aux term is added to the returned (combined)
`loss`; `Trainer.evaluate()` reads `ce_loss` instead of `forward()`'s return value, so `val_loss`
stays directly comparable to every other wave's (docs/EXPERIMENTS.md's noise-floor convention
only ever measured plain CE). `train_step`'s reported `train_loss` is UNCHANGED (still the
combined objective) — matches the existing z-loss precedent, where `train_loss` has always meant
"whatever was actually optimized," not pure CE. Added a regression test
(`test_evaluate_val_loss_excludes_aux_terms`) proving `val_loss` is identical regardless of
`aux_loss_weight`. The two buggy run folders (fresh this session, no notes.md/verdict ever
written, registry rows still auto-generated placeholders) were deleted rather than kept as
confusing superseded duplicates in the lab record, then re-run clean with the fix in place —
correct per the system's "own-session scratch work" carve-out to CLAUDE.md's don't-delete-runs
rule, not a precedent for deleting any run with an actual conclusion attached.

**Results (post-fix, real numbers — see `docs/results/ablation_log.md` for the full writeup):**
DeepSeekMoE reproduces its headline win at S-tier (-0.09 vs control on both balancing methods,
>4x noise floor) — more total capacity via fine-grained experts genuinely helps at matched
active params/token. The two balancing methods are statistically tied on final quality (0.008
apart) but differ measurably in balancing SPEED (aux_loss's gradient signal balances by step
~200; bias_free's bounded per-step nudge takes until step ~800-1000) — a real, clean
reproduction of the mechanistic tradeoff DeepSeek-V3's paper describes. MTP is not
distinguishable from noise at this scale/token budget (+0.017, at the noise floor's edge),
though the extra head does demonstrably learn its own (harder) task.

**Revisit if:** MTP is worth a follow-up sweep (`loss_weight`, `n_predict_tokens>1`) at a larger
tier/token budget before ruling it out of the phase-9 capstone recipe; DeepSeekMoE is a strong
capstone candidate if the L-tier's total-parameter budget can absorb ~2x growth for the FFN
layers. Any future wave computing a NEW auxiliary/regularization loss term inside `GPT.forward`
should follow this fix's pattern (store pure CE separately, keep it out of the eval metric)
rather than re-deriving the lesson.
