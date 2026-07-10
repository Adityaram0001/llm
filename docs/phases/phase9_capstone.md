# Phase 9 — Capstone: the 100M hero run + final report

**Goal:** one L-tier (~100M) pretrain using the best recipe discovered in phase 5, fine-tuned
via the phase-8 pipeline, fully evaluated; then the write-up that consolidates everything
learned.
**Effort:** 1 session to launch + a 1–3 day background run + 1 session to finish.

## Steps

1. **Recipe freeze:** from `docs/results/recipe.md`, assemble `configs/model_l_hero.yaml` +
   `configs/train_l_hero.yaml`. Every choice must cite a decision (D-xxx) or a run_id verdict —
   this config is the project's thesis statement.
2. **Data:** decide (with user) books+dictionary multi-epoch vs +supplement (D-006). Target
   ≥1B tokens seen. Re-verify tokenized shards.
3. **Dry run:** 30-min L-tier smoke at hero settings; verify tok/s, memory headroom (<12GB RSS),
   checkpoint size, ETA math. Present ETA to user for go/no-go.
4. **Hero run — pick venue with the user (D-010):**
   - **Rented RTX 5090 (recommended):** follow `docs/CLOUD.md` end-to-end (sync up, tmux,
     wandb online, periodic sync_down, STOP pod). Expect overnight-ish, ~$10–20. If it's the
     user's first rental, do the $1 practice run first.
   - **Local Mac:** `caffeinate -is python scripts/train.py ...` in terminal, other apps
     closed; expect 1.5–3 weeks at the D-006 token budget (D-008) — only sane with a reduced
     budget or WSD checkpoints along the way.
   WSD schedule recommended either way (mid-run decay branches = free intermediate models).
   Resume-on-interrupt is battle-tested (P4) — interruptions are fine.
5. **Post:** full eval suite; SFT + (optional) DPO on top; chat demo; side-by-side vs the
   S-tier baseline from phase 4 (the "how far we came" table).
6. **Final report — `docs/results/final_report.md`:**
   - project narrative; corpus & tokenizer choices;
   - the ablation league table (all registry verdicts condensed);
   - hero-run curves & evals; what the dictionary did for the model;
   - "what I'd do differently"; open questions → candidate future experiments;
   - appendix: complete decision log reference.
   Optionally condensed into a blog-post/portfolio piece (great interview artifact — ties to
   the user's job-search context).

## Exit criteria
Hero checkpoint + eval JSON archived; report done; PROGRESS all-green; M5. Project v1 complete.
Afterwards the lab remains open: new papers → new Wave letters in phase 5.
