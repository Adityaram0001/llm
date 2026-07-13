# 20260713_p5_s-wave-d-constant — constant+warmup schedule (no decay)

**Hypothesis:** warmup then flat lr forever should train comparably to cosine mid-training but
end worse (no decay tail to settle into a sharper minimum).

- **Result:** val_loss **3.4303** (ppl 30.88), **-0.0674 vs control's cosine** — real, past the
  0.015-0.02 (D-035) noise floor, and (surprisingly) BETTER than cosine, not worse.
- **Reading:** this only makes sense alongside WSD's result — the hierarchy is
  **WSD (-0.1213) > constant (-0.0674) > cosine (control)**. Some decay (WSD, at the very end)
  beats no decay (constant), which in turn beats early/continuous decay (cosine). The lesson:
  decaying the LR matters, but *when* you decay matters just as much — spreading the decay across
  the whole run (cosine) is worse than not decaying at all here, let alone a short decay at the
  end (WSD).
- **Conclusion:** REAL result, and the wave's clearest illustration of *why* WSD/late-decay
  schedules are gaining adoption over plain cosine. Also serves as the shared "stable phase"
  checkpoint for the WSD multi-budget bonus demo (see wsd_branch_short/long) — its final step-1500
  checkpoint (still at peak lr, undecayed) is what both fork runs resume from.
