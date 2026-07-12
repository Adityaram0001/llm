# EXPERIMENTS — protocol & registry

The lab record. Rules here make runs comparable months later.

## One run = one folder

```
experiments/<run_id>/
├── config.yaml        # EXACT resolved config used (auto-dumped by the trainer)
├── metrics.jsonl      # one JSON object per log step (see schema below)
├── notes.md           # hypothesis → observation → conclusion (3 lines minimum)
├── samples/           # generated text at checkpoints (step_001000.txt …)
└── ckpt/              # latest.pt, best.pt
```

`run_id` = `YYYYMMDD_p<phase>_<slug>` e.g. `20260801_p5_rmsnorm-vs-layernorm-a`.

## metrics.jsonl schema (per logged step)

```json
{"step": 1200, "tokens_seen": 39321600, "train_loss": 3.41, "val_loss": 3.62,
 "lr": 0.00028, "grad_norm": 0.71, "tokens_per_sec": 5100, "mem_gb": 8.9,
 "elapsed_s": 7712}
```
`val_loss` present only on eval steps. Add fields freely; never rename existing ones.

## registry.csv schema (one row per run, append-only)

```
run_id, date, phase, tier(S/M/L), params_M, baseline_run(or "-"), variable_changed,
tokens_trained_M, final_val_loss, final_ppl, wall_hours, wandb_url, verdict(one sentence)
```

## Ablation protocol (the scientific method part)

1. **Name the baseline.** Every ablation references a baseline `run_id` with identical config
   except ONE variable.
2. **Write the hypothesis first** in `notes.md` *before* the run ("RMSNorm will match LayerNorm
   loss at ~5% higher tokens/sec").
3. **Same seed, same data order** as baseline unless seed-variance is the thing being measured.
4. **Judge on val loss at equal tokens-seen AND equal wall-clock** — a technique can win on one
   axis and lose on the other; that distinction is a core lesson of this project.
5. **Seed noise floor:** early in phase 5, run the S-tier baseline with 3 seeds; the spread
   defines the "not real unless bigger than this" threshold quoted in every verdict.
6. **Conclusion in notes.md + verdict in registry.csv** right after the run, while it's fresh.

## Seed noise floor (phase 5, established 2026-07-12)

S-tier baseline recipe (`configs/train_s_baseline.yaml`, D-021), 3 seeds, 1500 steps / 98.3M
tokens each: seed 1337 (`20260711_p4_s-baseline`) val_loss 3.5037; seed 1338
(`20260712_p5_s-seed-1338`) 3.4970; seed 1339 (`20260712_p5_s-seed-1339`) 3.5121.

**mean = 3.5043, std = 0.0062, spread (max-min) = 0.0150.**

**Rule for every Wave A-G verdict:** a val_loss delta from its named baseline smaller than
~0.015-0.02 is within the seed-noise floor and must be reported as "not distinguishable from
noise," not as a real effect. Deltas should be compared against this floor explicitly in each
run's notes.md conclusion, not just eyeballed. See D-035.

## Comparison studies

`notebooks/compare_runs.ipynb` (built in phase 4) loads registry.csv + any set of
metrics.jsonl files and renders: loss-vs-tokens, loss-vs-wallclock, tokens/sec bars,
memory bars. Every phase-5 study ends with a saved figure in `docs/results/`.
