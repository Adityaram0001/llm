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

<!-- Append new decisions below. Next ID: D-019 -->
