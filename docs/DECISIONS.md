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

<!-- Append new decisions below. Next ID: D-008 -->
