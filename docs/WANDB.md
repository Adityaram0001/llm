# WANDB — dashboard setup & how this project uses it

> Companion to `docs/EXPERIMENTS.md` (the actual source-of-truth protocol: `registry.csv` +
> local `metrics.jsonl`, per D-005). wandb is a *visualization layer* on top of that data, not a
> replacement for it — if wandb.ai ever goes down or the free tier runs out, nothing about this
> project's record-keeping breaks.

## What wandb is for here, and what it isn't

- **Source of truth stays local**: every run writes `config.yaml` + `metrics.jsonl` +
  `notes.md` into `experiments/<run_id>/` regardless of wandb mode (D-005). `registry.csv` +
  `notebooks/05_compare_runs.ipynb` already do cross-run comparison from these files alone —
  no wandb account needed for that at all.
- **wandb adds**: a hosted, point-and-click dashboard for overlaying loss curves across many
  runs, live-updating plots while a cloud run is training, and system metrics (GPU util/power/
  VRAM) charts it collects automatically. Nice-to-have UX on top of data that already exists
  locally either way.
- Every run has *always* been logging to wandb — just in **offline mode** by default (D-009):
  data was written to `experiments/<run_id>/wandb/offline-run-*/` on whatever machine ran the
  training, but never uploaded. Nothing was "missing" before this setup, it just wasn't synced.

## Credentials

`WANDB_API_KEY` + `WANDB_ENTITY` live in the root `.env` (gitignored — same file as the R2
credentials, D-026). **Never commit `.env`** or paste the API key into any tracked file,
including this one. `.env.example` documents the two variable names with empty placeholders.

The **project name is NOT an env var** — it's `TrainConfig.logging.wandb_project`
(`src/llmlab/train/config.py`), defaulting to `"llm-lab"` (D-005). This is deliberate: every run
ever logged (Mac and cloud, phases 4–5) already used this project name, so keeping it
config-driven instead of env-driven means it can't silently drift to a different project on a
new machine just because someone's `.env` has a different value. All runs land in one place:
`https://wandb.ai/<WANDB_ENTITY>/llm-lab`.

## Online vs. offline: how to choose per run

D-009's offline-by-default stands — unchanged. To stream a **specific** run live instead:

```bash
python scripts/train.py --config configs/train_s_wave_f_moe.yaml --wandb-online
```

`--wandb-online` (added this session, `scripts/train.py`) overrides that one run's
`wandb_mode` to `"online"` at launch — it does not change the config file or any other run.
Equivalently, set `logging: {wandb_mode: online}` directly in a YAML if you want a specific
config to always stream (e.g. a long unattended overnight run where you want to watch it from
the Mac's browser without babysitting SSH).

**Requirement:** `WANDB_API_KEY` must be present in the environment wherever `train.py` runs.
For cloud runs, that means the pod's own `.env` needs the key — `scripts/cloud/wandb_sync.sh`
(below) pushes the Mac's `.env` to the pod as a side effect, or `scp .env` it up manually before
a run (same pattern `gpuhub_setup.sh` already uses for the R2 credentials).

**Recommended pattern for cloud waves going forward:** use `--wandb-online` for any run you
expect to actually watch live (e.g. the first run of a new wave, or a long L-tier run later);
plain offline is fine for a batch of short, unattended S-tier ablation runs where you'll review
the notebook afterward anyway — no need to babysit 6 five-minute runs in a browser tab.

## Syncing offline runs after the fact

`scripts/cloud/wandb_sync.sh` — Mac-side wrapper, run any time:
1. Pushes the Mac's current `.env` to the pod (`scp`, so the pod always has the latest
   `WANDB_API_KEY`).
2. SSHes in and runs `wandb sync` on every `experiments/*/wandb/offline-run-*` directory found
   on the pod's disk.
3. Idempotent — `wandb sync` skips runs it's already pushed, safe to re-run after every wave.

This only syncs what's physically present as an offline run directory. `experiments/**/wandb/`
is gitignored (same as `ckpt/`), so if a pod is terminated before syncing, that specific run's
wandb history is gone for good — the `metrics.jsonl`/`config.yaml`/`notes.md` triplet is NOT
affected (separately committed to git / archived to R2 per D-041), only the wandb dashboard
copy. **Run `wandb_sync.sh` before shutting down a pod that did offline-mode training.**

**First sync (2026-07-16):** all 33 offline runs found on the `singapore-b:25864` pod were
pushed and independently verified via `wandb.Api()` — 33/33 present, all `state=='finished'`.
The first attempt looked successful (CLI printed "done." for every run) but had actually failed
silently server-side due to a wrong `WANDB_ENTITY` — see D-042 for the full story. **Lesson:
always verify a sync against `api.runs(...)`, never trust the CLI's own "done." alone.**

## Where this fits with R2 (D-041)

R2 and wandb solve different problems and don't overlap:

| | R2 (`experiments/` archive) | wandb |
|---|---|---|
| Holds | checkpoints (slim by default) + config/metrics/notes | loss curves, system metrics, live view |
| Durability | permanent object storage, survives any pod being terminated | depends on free-tier retention; not the archival copy |
| Needed to resume/reload a model | yes | no |
| Needed to compare runs visually | no (the notebook works from local files) | yes, if you want the hosted dashboard specifically |

## Quick links

- Dashboard: `https://wandb.ai/<WANDB_ENTITY>/llm-lab` (fill in your entity from `.env`)
- D-009 (offline-by-default decision), D-041 (R2 checkpoint archival), D-042 (this setup),
  D-043 (a related cloud-throughput gotcha found while working on this — see below)
