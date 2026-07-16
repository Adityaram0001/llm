# 20260716_p5_scaling-5m — Wave G mini scaling law, N=5.00M

**Role:** Point 1/4 of the mini scaling law (D-045). `configs/model_scaling_5m.yaml`
(d_model=128, n_layers=15, n_heads=2, head_dim=64 fixed — same conventions as S/M/L-tier).
Fixed 200M-token budget (3050 steps @ 65,536 tok/step ≈ 199.9M tokens) over the SAME 17.66M-
token books+dictionary pool used by every other S-tier ablation (~11.3 epochs of repetition —
identical repetition at every N in this study, so only param count varies). lr held fixed at
1e-3 across all 4 sizes (NOT muP-retuned per size — a real caveat, see D-045).

- **Result:** val_loss=3.4031 (ppl 30.06) at step 3050 — still monotonically improving at the
  end of the budget (no repetition-driven overfitting yet at this capacity), so best=final.
- **Conclusion:** smallest point of the scaling curve — see D-045 for the fitted power law and
  the larger models' very different (overfitting) behavior.
