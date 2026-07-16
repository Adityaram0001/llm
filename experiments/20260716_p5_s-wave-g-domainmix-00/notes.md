# 20260716_p5_s-wave-g-domainmix-00 — Wave G domain-mix ablation, control (0%)

**Role:** Control point for RW-4's domain-mix dose-response curve (D-045). General
books+dictionary corpus only (`train.bin`/`val.bin`), no `domain_books` source mixed in.
Same model/lr/seed as every other S-tier ablation, but a REDUCED fixed token budget of
49.15M tokens (750 steps @ 65,536 tok/step), not the usual ~98.3M — the new 62-book
finance/self-help/wisdom domain pool (D-045) is only 6.76M raw train tokens, and the 50%-share
point at the standard ~98.3M budget would need >7x repetition of that pool, blowing past the
phase-5 spec's own "domain repetition ≤4 epochs" design constraint. **val_loss here is NOT
directly comparable to the D-035 noise floor or any other wave's number (different budget) —
only compare the 4 domain-mix runs against each other.**

- **Result:** val_loss=3.9800 (ppl 53.52).
- **Conclusion:** valid control for the domain-mix sweep — see the 10/25/50% runs' notes and
  D-045 for the dose-response verdict.
