# 20260713_p5_s-wave-d-gradclip_off — grad_clip effectively disabled (1e6)

**Hypothesis (per the phase-5 spec):** removing grad clipping should cause a visible loss/grad
spike — "watch it spike."

- **Result:** val_loss 3.5192 (ppl 33.76), **+0.0215 vs control** — just past the 0.015-0.02 (D-035) noise
  floor: a real but SMALL degradation, not a spike/blowup.
- **What actually happened (see wave_d_optimizers_schedules.png panel d):** `grad_norm` peaks at
  ~5.51 at step 0 for BOTH the clipped and unclipped run — `clip_grad_norm_` always returns the
  PRE-clip norm regardless of whether clipping is applied, so the logged metric can't show a
  difference by construction. The real difference is in what happens to the loss AFTER that big
  early gradient: gradclip_off's train_loss is consistently ~0.02-0.1 higher than control's at
  every early checkpoint (step 10-70), a small but persistent gap that tracks the final val_loss
  delta — not a single dramatic event, just steady mild noise from letting the occasional large
  gradient through unscaled.
- **Conclusion:** REAL but UNDRAMATIC — the opposite of what the spec's "watch it spike" framing
  predicted. At S-tier/15-layer/pre-norm depth with a 30-step warmup, this architecture is
  already stable enough that grad_clip=1.0 rarely binds hard, so removing it costs a little
  quality but doesn't blow up training. A more aggressive stress test (higher lr, no warmup, or a
  much longer run where a rare large gradient has more chances to land) would likely be needed to
  produce the dramatic spike the spec describes — worth flagging as a follow-up if a future wave
  wants that demo specifically.
