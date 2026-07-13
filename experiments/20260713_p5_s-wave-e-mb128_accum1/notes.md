# 20260713_p5_s-wave-e-mb128_accum1 — micro-batch/accum equivalence, mb=128 accum=1

**Hypothesis:** same effective batch as the control (128 seqs/step) via the opposite
factorization — one micro-step per optimizer step, zero grad accumulation. Loss should match
the control; speed should be at least as good since there's no accumulation-loop overhead at all.

- **Quality:** val_loss 3.5017 (ppl 33.17), **+0.0040 vs control (3.4977)** — within noise.
  Confirms equivalence again, from the opposite direction of `mb32_accum4`.
- **Speed:** ~525,106 tok/s — the **fastest of the three factorizations tested**, ~15% faster
  than the control (mb=64/accum=2) and more than 2x faster than `mb32_accum4` (mb=32/accum=4) —
  despite all three doing identical total FLOPs for identical loss. Fewer, bigger micro-steps
  amortize kernel-launch/Python-loop overhead best.
- **Conclusion:** combined with `mb32_accum4` and the control, this closes the micro-batch/accum
  equivalence check cleanly: **loss is factorization-invariant, wall-clock is not** — always
  prefer the largest micro-batch that fits in memory (D-018's calibration-tool guidance),
  minimizing grad_accum rather than treating it as a free knob.
