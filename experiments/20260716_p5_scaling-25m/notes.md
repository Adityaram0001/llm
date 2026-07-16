# 20260716_p5_scaling-25m — Wave G mini scaling law, N=24.79M

**Role:** Point 3/4. `configs/model_scaling_25m.yaml` (d_model=320, n_layers=16, n_heads=5).
Same 200M-token budget/pool/lr convention as `20260716_p5_scaling-5m` — see its notes.

- **Result:** val_loss bottoms out at **3.1655 at step 2400 (~157M tokens, ~8.9 epochs)**, then
  RISES to 3.1789 by the final step 3050 (~11.3 epochs) — a +0.0134 "overfit gap" between best
  and final, while train_loss keeps falling all the way to 2.738. This is the first point in the
  scaling curve where the model's capacity is large enough to start memorizing the repeated
  17.66M-token pool before the fixed token budget runs out.
- **Conclusion:** the scaling-law fit (D-045) uses the BEST (early-stopped) value, not the final
  one, for exactly this reason — see the 50M run for a much more severe version of the same
  effect, and D-045 for the full writeup tying this to the project's existing Muennighoff-
  ceiling concept (RW-1/D-015).
