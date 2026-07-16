# 20260716_p5_scaling-50m — Wave G mini scaling law, N=50.40M

**Role:** Point 4/4, the largest model in this study. `configs/model_scaling_50m.yaml`
(d_model=384, n_layers=25, n_heads=6). Same 200M-token budget/pool/lr convention as
`20260716_p5_scaling-5m` — see its notes.

- **Result:** val_loss bottoms out at **3.1701 at step 1650 (~108M tokens, only ~6.1 of the
  ~11.3 epochs the budget allows)**, then RISES sharply to 3.2789 by the final step 3050 — a
  +0.1088 "overfit gap", by far the largest in the study, while train_loss keeps falling all the
  way to 2.285 (the lowest train loss of any of the 4 sizes). The model has enough capacity to
  memorize the repeated 17.66M-token pool well before half its remaining budget is spent.
- **Conclusion: the headline finding of this wave.** At a FIXED, small, heavily-repeated token
  budget, bigger models overfit both faster and harder — a clean, concrete illustration of why
  Chinchilla-style scaling laws assume fresh (not repeated) tokens, and a direct empirical tie to
  this project's own already-established "Muennighoff ceiling" concept (RW-1/D-015: repeated-data
  returns diminish past ~4 epochs). The scaling-law fit (D-045) uses this run's BEST value
  (3.1701), not the final one (3.2789) — using the final number would have made 50M look worse
  than 25M and badly misrepresented the true capacity-vs-loss relationship.
