# results figures & reports land here (ablation_log.md, recipe.md, final_report.md)

- `cloud_gpu_benchmarks.csv` — raw `find_batch_size.py` sweep data (every micro_batch × tier ×
  seq_len × GPU data point, not just the summarized sweet-spot numbers) from the gpuhub RTX
  4080/5090 capacity comparison (D-030/D-031/D-032). Load with pandas for any follow-up analysis
  instead of re-running the sweeps. Full narrative/reasoning:
  `docs/learnings/20260712_gpuhub-rtx4080-capacity.md`.
