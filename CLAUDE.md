# LLM-Lab — Rules for Claude Sessions

This project is a **learning lab**: build a ~100M-parameter GPT-style LLM from scratch on a
MacBook Pro M4 (16GB RAM, 512GB SSD, macOS, Apple GPU via PyTorch MPS), then use it as a
testbed to implement and compare training-optimization techniques from research papers.
**The user's learning is the product** — a working model is the by-product.

## Session start protocol (ALWAYS do this first)

1. Read `PROGRESS.md` — current phase, what's done, what's next, open blockers.
2. Read the active phase spec in `docs/phases/` (PROGRESS.md names it).
3. Skim the last ~5 entries of `docs/DECISIONS.md` so you don't re-litigate settled choices.
4. Only then start working. Do NOT redesign the project structure or re-decide logged decisions.

## Session end protocol (ALWAYS do this before finishing)

1. Update `PROGRESS.md`: check off completed items, note in-progress state precisely enough
   that a fresh session (with zero conversation context) can resume.
2. Log every non-obvious choice made this session in `docs/DECISIONS.md` (format is in that file).
3. If any training/eval run happened, ensure it is registered in `experiments/registry.csv`
   and its run folder is complete (config + metrics + notes).

## Teaching mode (core requirement)

The user is an experienced data scientist (4 yrs; knows sklearn/stats/basic transformers,
has fine-tuned LLMs) but is learning LLM internals hands-on. Therefore:

- When implementing a concept, **explain the what/why/trade-offs briefly as you go**
  (2–6 sentences, not essays). Name the paper a technique comes from.
- At every decision point listed in the phase spec, present the options + trade-offs, make a
  recommendation, let the user choose (or apply the logged default), and record it in DECISIONS.md.
- Prefer readable code over clever code. Type hints, docstrings on public functions,
  comments only where the math/logic is non-obvious.
- Each phase spec has "Learning checkpoints" — questions the user should be able to answer
  after the phase. Point them out when relevant.

## Hardware & memory rules (16GB RAM — this is tight)

- **Two targets, one codebase**: primary = this Mac (MPS); burst = rented Linux/NVIDIA pods
  (RTX 5090) for M/L-tier runs — playbook in `docs/CLOUD.md` (D-010). ALL code must be
  device-agnostic: device only via `llmlab.utils.get_device()` (cuda > mps > cpu), mixed
  precision only via `llmlab.utils.autocast_ctx(device)`, `torch.mps.*`/`torch.cuda.*` calls
  guarded by `is_available()`, checkpoints loaded with `map_location=get_device()`,
  DataLoader `num_workers`/`pin_memory` as config keys (0/False on Mac). Read CLOUD.md's
  portability rules before writing any training code.
- Training/long jobs = **Python scripts run from terminal** (user closes other apps).
  Exploration/visualization/teaching = **Jupyter notebooks**. Never train seriously in a notebook.
- Local device: `mps`, bf16 autocast via `autocast_ctx`; keep optimizer states fp32.
  Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in scripts (harmless on Linux).
- Never load the full corpus into Python lists. Tokenized data lives in `data/tokenized/` as
  `uint16` numpy memmap files; the DataLoader reads slices.
- Default training shapes unless a spec says otherwise: seq_len 512, micro-batch that keeps
  RSS under ~10GB, gradient accumulation for larger effective batch. Print/track memory
  (`psutil`, `torch.mps.current_allocated_memory()`) in every training script.
- Checkpoints: keep `latest.pt` + `best.pt` per run, plus milestone snapshots only if the spec
  asks. 512GB disk — don't hoard optimizer states in old checkpoints.
- `torch.compile` on MPS is unreliable — treat it as an optional experiment, never a dependency.

## Experiment discipline (core requirement)

- Everything is **config-driven**: a run = one YAML in `configs/` (or a variant recorded in the
  run folder). No hyperparameters hard-coded in scripts.
- One run = one folder: `experiments/<run_id>/` with `config.yaml`, `metrics.jsonl`,
  `notes.md`, checkpoints. `run_id` format: `YYYYMMDD_<phase>_<short-slug>` (e.g. `20260712_p5_rope-vs-learned`).
- Every run gets a row in `experiments/registry.csv` (schema in `docs/EXPERIMENTS.md`).
- Seed everything (`llmlab.utils.set_seed`). Log the seed. Ablations change ONE variable
  vs a named baseline run.
- Tracking: Weights & Biases (project `llm-lab`) **plus** local `metrics.jsonl` always
  (wandb can be offline; local files are the source of truth).

## Environment

- venv at `.venv/` (created by `scripts/setup.sh`), Python 3.11+. All deps in `requirements.txt` —
  if you add a package, add it there with a pinned major version and note why in DECISIONS.md.
- Package code lives in `src/llmlab/` (installed editable). Entry-point scripts in `scripts/`.
  Notebooks import from `llmlab`, never duplicate its code.

## Data & licensing rules

- Books: public-domain only (Project Gutenberg). Strip Gutenberg headers/footers.
- Dictionary: use a public-domain/free source (Webster's 1913 via GCIDE, or Wiktionary extract).
  **Do not** scrape Oxford dictionaries — copyrighted. (Logged: DECISIONS.md D-003.)
- Q&A / instruction data generation ("data factory", `tools/data_factory/`): human-in-the-loop
  batch workflow with DeepSeek web chat — the user pastes prompts and saves replies manually;
  scripts only prepare batches and parse/validate outputs. **Never build or run browser
  automation against DeepSeek's web UI** (ToS violation; logged D-004). The DeepSeek API is a
  supported optional backend if the user enables it.

## Change management (when new learning invalidates old work)

Decisions WILL be revised as the user learns — that's the point of the project. The mechanism:

1. A revised decision gets a NEW entry in DECISIONS.md (never edit the old one) that names the
   entry it supersedes, plus an **Impacts:** line listing affected artifacts/phases.
2. Every impacted artifact becomes a row in the **Rework queue** in PROGRESS.md
   (`RW-N | what | why (D-xxx) | fix in phase | status`). Completed phases are NOT reopened
   ad hoc — rework rows are picked up at the start of whichever phase they're tagged to.
3. Sessions must check the rework queue at session start (it's part of PROGRESS.md) and pick up
   rows tagged for the active phase before new work.
4. Never silently redo old work: if you find yourself changing a completed phase's artifact,
   there must be a D-entry + RW-row explaining why.

## Discussion sessions (learning-capture protocol)

The user will open chats purely to ask questions, clarify concepts, and reflect — no
implementation. Trigger: the user opens with **"Discussion session: <topic>"** (any phrasing
that clearly says discussion/doubts/concepts counts).

Rules for such sessions:
- Read PROGRESS.md + relevant phase spec/decisions first, as usual — answers must reflect the
  project's actual state and numbers, not generic theory.
- Teaching mode, depth over speed. Diagrams/tables/worked arithmetic encouraged. It's fine to
  reference code, but **change no code and no specs** — if the discussion uncovers needed
  changes, log a decision + rework-queue row instead (that's the paper trail).
- **Before the session ends, write a learning note** to `docs/learnings/YYYYMMDD_<slug>.md`:
  the user's questions, the explanations that clicked (with the actual numbers/math used),
  misconceptions corrected, takeaways, and links to related D-entries/runs/papers. Style:
  written for the user's future revision, not a chat transcript.
- Add one line to `docs/learnings/INDEX.md` (newest first). If any decision/rework rows were
  spawned, update DECISIONS.md/PROGRESS.md per the change-management protocol above.

## Things NOT to do

- Don't install CUDA-only packages (flash-attn, bitsandbytes, deepspeed, xformers) — Apple Silicon.
- Don't start multi-hour training runs without telling the user the estimated time and getting a go.
- Don't delete or overwrite run folders in `experiments/` — they are the lab record.
- Don't "upgrade" settled decisions (tokenizer choice, base config, etc.) mid-phase; propose in
  notes and let the user decide.
- Don't pull huge datasets (>2GB) without asking; disk and bandwidth budgets matter.
