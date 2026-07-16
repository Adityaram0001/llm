# 20260716_p5_s-wave-g-domainmix-25 — Wave G domain-mix ablation, 25% share

**Role:** `sources: books_dict weight=0.75, domain_books weight=0.25`. Same budget/seed/model
and general-val eval convention as the 0%/10% runs — see
`20260716_p5_s-wave-g-domainmix-00`'s notes for the full setup rationale.

- **Result:** val_loss=4.0549 (ppl 57.68), **+0.0749 vs the 0% control**.
- **Qualitative check:** `samples/step_000700.txt` already shows finance/economics-inflected
  vocabulary ("Export of the State", currency/trade phrasing, historical dates) at only 49M
  training tokens — the domain mix is visibly steering generation, not just moving a metric.
- **Conclusion:** continues the monotonic degradation trend (0% < 10% < 25% < 50%) — see D-045.
