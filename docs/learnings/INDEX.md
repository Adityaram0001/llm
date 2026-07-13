# Learnings index

One line per note, newest first. Notes are written at the end of discussion sessions
(see CLAUDE.md "Discussion sessions") — dense, revision-friendly, with the project's real
numbers. Not chat transcripts.

- [2026-07-13 — Wave D deep dive: Muon's Newton-Schulz orthogonalization + why its edge narrows but never closes, the WSD>constant>cosine schedule hierarchy explained, the WSD multi-budget checkpoint-fork demo, why grad-clip-off didn't spike, Lion/batch-size confounds flagged honestly, and whether any of it breaks re-running older waves](20260713_wave-d-optimizers-schedules.md)
- [2026-07-12 — Sequence length vs. token count vs. model size: what each buys you, minimum config per learning goal mapped to phase 5's waves, why the capstone's chat-context need is a separate decision](20260712_model-config-strategy.md)
- [2026-07-12 — gpuhub RTX 4080 capacity: measured throughput/memory sweeps across tiers & seq_len, why tok/s regresses past micro_batch=32, cost estimates for S/M/L runs on the $0.25/hr tier](20260712_gpuhub-rtx4080-capacity.md)
- [2026-07-12 — R2 bucket file walkthrough: tokenizer artifacts vs tokenized data, docstarts/boundary-crossing default, Chinchilla margin math behind FineWeb-Edu](20260712_r2-data-files-walkthrough.md)
- [2026-07-11 — GPU choice ($/FLOP not VRAM), batch calibration vs dynamic adjustment, 32k-vocab v2 parking, domain data mixing math](20260711_gpu-vocab-datamix.md)
- [2026-07-11 — Parameter allocation: embeddings vs "active" params, and how model size couples to data budget](20260711_parameter-allocation.md)
