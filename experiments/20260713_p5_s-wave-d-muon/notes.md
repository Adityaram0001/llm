# 20260713_p5_s-wave-d-muon — Muon optimizer (Newton-Schulz orthogonalized momentum)

**Hypothesis:** Muon (Jordan '24) should train faster than AdamW on the 2D hidden matrices
(attn/ffn projections), per the nanoGPT speedrun results -- AdamW still handles embeddings/norms
(muon_lr=0.02, aux AdamW lr=1e-3).

- **Result:** val_loss **3.3432** (ppl 28.31), **-0.1545 vs control** — more than 10x the noise
  floor (0.015-0.02 (D-035)). The single biggest effect in all of Wave D.
- **Trajectory:** the gap is largest early (tokens@~6M: -0.4, see docs/results/
  wave_d_optimizers_schedules.png) and *narrows* over training (-0.267 @ step500, -0.185 @
  step1000, -0.155 final) but never closes — Muon is consistently ahead at every checkpoint, not
  just a lucky final read.
- **Conclusion:** REAL, ROBUST, BEST OF THE WAVE. Reproduces the nanoGPT speedrun's headline
  claim at 10M params: Muon is primarily a *convergence-speed* accelerator (biggest edge early),
  which matches its "gets you to a target loss faster" framing better than "raises the asymptote"
  framing. Worth being the flagship recipe recommendation for phase 9 if training-time is the
  binding constraint. Caveat: this used one un-tuned `muon_lr=0.02` guess (Jordan's commonly-cited
  default) — a proper muon_lr sweep is a natural follow-up, not done this session (time budget).
