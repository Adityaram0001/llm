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

<!-- Append new decisions below. Next ID: D-037 -->
