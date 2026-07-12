# 20260712_p5_s-seed-1339 — seed-noise study, run 3/3 (noise floor computed here)

**Hypothesis:** same as `20260712_p5_s-seed-1338` — identical config to
`20260711_p4_s-baseline` except `seed=1339`, should land within a small band of the other two
seeds' val_loss.

**Observation:** final val_loss 3.5121 (ppl 33.52), ~12.7 min wall-clock on the RTX 5090 (same
config/tokens/step as baseline and seed-1338, ~11x-53x faster than the Mac baseline's 2.4hr
depending on which GPU throughput number is compared).

**Noise floor across all 3 seeds (1337/1338/1339):**

| seed | final val_loss | final ppl |
|------|-----------------|-----------|
| 1337 (baseline) | 3.5037 | 33.24 |
| 1338 | 3.4970 | 33.02 |
| 1339 | 3.5121 | 33.52 |

mean = 3.5043, std = 0.0062, **spread (max-min) = 0.0150**.

**Conclusion:** the S-tier baseline recipe (D-021) is reproducible within ~0.015 val_loss across
seeds at this token budget (98.3M tokens). Per `docs/EXPERIMENTS.md`'s protocol, this spread is
now the noise floor every later ablation verdict must clear: **a Wave A-G result whose val_loss
delta from its baseline is smaller than ~0.015-0.02 should be reported as "within noise," not as
a real effect.** Logged as D-035; full noise-floor entry recorded in `docs/EXPERIMENTS.md`.
