# 20260716_p5_s-wave-g-domainmix-10 — Wave G domain-mix ablation, 10% share

**Role:** `sources: books_dict weight=0.9, domain_books weight=0.1` — 10% of every training
batch (in expectation) is drawn from the 62-book finance/self-help/wisdom pool (D-045), 90%
from the general books+dictionary corpus. Same 49.15M-token budget/seed/model as the control
(`20260716_p5_s-wave-g-domainmix-00`). Eval is always against the GENERAL val set (books+
dictionary) — measures the cost to general quality, not domain quality (a domain-specific probe
is phase 6 work, not built yet).

- **Result:** val_loss=4.0153 (ppl 55.44), **+0.0353 vs the 0% control**.
- **Conclusion:** real, if modest, degradation of general val loss even at just 10% domain
  share — the finance/self-help register is different enough from the general corpus's academic
  philosophy + dictionary register that mixing it in has an immediate, measurable cost on the
  general held-out set. First point of a clean 4-point monotonic dose-response curve — see
  D-045 and the 25%/50% runs.
