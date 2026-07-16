# 20260717_p6_s-p6-baseline-milestones

**Hypothesis:** identical recipe to `20260711_p4_s-baseline` (D-021: seed 1337, same lr/schedule/
data/effective batch), run purely to capture named `step_000150.pt`/`step_000750.pt`/
`step_001500.pt` snapshots (early/mid/final) for `notebooks/08_eval_deep_dive.ipynb` — the
original baseline run only ever kept `latest.pt`/`best.pt`. Since the loader is stateless in
`(seed, step)` (D-021) and the config is otherwise unchanged, this should also reproduce the
original baseline's loss curve closely — an incidental reproducibility check.

**Observation:** ran on the RTX 5090 (singapore-b:25864) at `micro_batch=64` (D-043's cloud
sweet spot, vs the original Mac run's `micro_batch=16` — `grad_accum` halved to 2 to keep the
effective batch at 65,536 tokens/step identical). 1500 steps in 3m37s (~487K tok/s — matches
D-043's other 5090 S-tier numbers). Final val_loss **3.4954** vs the original baseline's
**3.5037** — a 0.0083 gap, well inside the D-035 seed-noise floor (spread 0.0150) despite the
different micro_batch/grad_accum factorization, consistent with D-040's "loss is
factorization-invariant" finding. Confirms the baseline recipe reproduces closely enough to
trust for the eval_deep_dive notebook's per-checkpoint trajectory.

**Conclusion:** milestone snapshots captured cleanly (`ckpt/step_000150.pt`, `step_000750.pt`,
`step_001500.pt`, plus the usual `latest.pt`/`best.pt`). `eval_results_step_*.json` (phase 6's
`scripts/evaluate.py --suite core`) written for all three; `eval_results.json` mirrors the final
(step 1500) checkpoint's results, matching every other run folder's convention. See D-046 for
the eval-suite build this snapshot trio supports, and `notebooks/08_eval_deep_dive.ipynb` for
the per-checkpoint discussion.
