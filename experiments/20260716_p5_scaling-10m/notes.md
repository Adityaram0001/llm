# 20260716_p5_scaling-10m — Wave G mini scaling law, N=9.71M

**Role:** Point 2/4. Reuses `configs/model_s.yaml` directly (the project's own S-tier baseline
architecture, d_model=192/n_layers=15/n_heads=3) rather than a new config — it already lands
within 2.9% of the 10M target and is the most battle-tested shape in the project. Same 200M-
token budget/pool/lr convention as `20260716_p5_scaling-5m` — see its notes for the full setup.

- **Result:** val_loss=3.2663 (ppl 26.21) at step 3050 — still monotonically improving at the
  end, best=final, same as the 5M point.
- **Conclusion:** see D-045 for the full 4-point curve and fit.
