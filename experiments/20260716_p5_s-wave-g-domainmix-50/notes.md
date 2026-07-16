# 20260716_p5_s-wave-g-domainmix-50 — Wave G domain-mix ablation, 50% share

**Role:** `sources: books_dict weight=0.5, domain_books weight=0.5` — the most aggressive share
tested. At this budget (49.15M tokens), the domain side is 24.6M tokens against a 6.76M-token
raw pool ≈ 3.6 epochs of domain repetition, still within the phase-5 spec's "≤4 epochs" design
constraint (D-045) that motivated this wave's smaller fixed budget in the first place.

- **Result:** val_loss=4.1442 (ppl 63.07), **+0.1642 vs the 0% control** — the largest gap in
  the sweep, >8x the D-035 noise floor (measured at a different budget, so only a rough scale
  reference, not a strict statistical bound here).
- **Conclusion:** confirms the specialization-vs-generality tradeoff is real and grows with
  domain share — a strictly monotonic 4/4-point dose-response curve (0.0 < 0.035 < 0.075 <
  0.164). See D-045 for the phase-9 recommendation on what share to actually use for a
  finance/wisdom-flavored capstone.
