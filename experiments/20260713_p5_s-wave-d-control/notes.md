# 20260713_p5_s-wave-d-control — Wave D control

**Role:** Wave D's own reference run, NOT a direct reuse of `20260711_p4_s-baseline`. Same
hyperparameters (lr=1e-3, betas=[0.9,0.95], wd=0.1, cosine, grad_clip=1.0, seed 1337), but
`micro_batch=64/grad_accum=2` instead of the Mac-tuned `16/8` (same 65,536 tok/step effective
batch, but the RTX 5090's measured S-tier sweet spot, D-030). Because the loader's stateless
`(seed, step)` sampling is keyed off `step * grad_accum + micro`, changing `grad_accum` changes
which data offsets land on which step even at an identical effective batch -- so this needed its
own control, same reasoning as Wave C's n_heads=4 control (D-038).

- **Hypothesis:** should land within noise of `p4_s_baseline`'s 3.5037 (only the batch/accum
  split differs, not any real hyperparameter).
- **Result:** val_loss 3.4977 (ppl 33.04) — within 0.015-0.02 (D-035) of `p4_s_baseline`. Confirms the
  batch/accum change alone doesn't move quality; ~4x fewer grad-accum loops made this run finish
  in ~1 minute vs the Mac-batch recipe's ~13-15 min on the same GPU.
- **Conclusion:** valid substitute control for every other Wave D run.
