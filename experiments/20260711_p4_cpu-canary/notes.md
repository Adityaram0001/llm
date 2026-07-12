# 20260711_p4_cpu-canary

**Hypothesis:** the training engine (`Trainer`, `scripts/train.py`, `MixedSourceLoader`) runs
unchanged with `device: cpu` -- the portability canary required by phase 4 deliverable 0b /
`docs/CLOUD.md` before trusting the same code on a rented CUDA pod.

**Observation:** 10 steps on the S-tier model + real books/dictionary corpus, forced to CPU.
Full pipeline exercised: warmup LR ramp, periodic eval (val_loss logged at steps 0 and 5),
text sampling (`samples/step_000000.txt`, `step_000005.txt`), `latest.pt`/`best.pt`
checkpointing, `metrics.jsonl` logging, registry row auto-appended. Train loss fell cleanly
9.71 -> 8.52 over 8 logged steps -- learning is happening, not just "runs without crashing."
Throughput: ~224 tokens/sec on CPU (vs D-008's ~20,800 tok/s MPS bench) -- expected, CPU eager
mode plus token-by-token `generate()` sampling is slow; this run is a correctness check, not a
speed benchmark.

**Conclusion:** portability canary passes -- deliverable 0b is done. No device-specific code
paths broke; `get_device()`/`autocast_ctx()` correctly no-op'd to CPU semantics. Safe to build
the real S-tier experiments next and to trust this same codebase on a rented CUDA pod later.
