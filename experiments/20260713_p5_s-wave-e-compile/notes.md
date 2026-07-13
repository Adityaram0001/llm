# 20260713_p5_s-wave-e-compile — torch.compile attempt (CUDA)

**Hypothesis:** CLAUDE.md flags torch.compile as unreliable on MPS and "an optional experiment,
never a dependency" — but this project's real training happens on the rented RTX 5090 (CUDA) for
anything beyond S-tier ablations, where torch.compile is much better supported. Attempt it here
and document whatever happens, per the phase-5 spec's explicit instruction for this axis.

- **Outcome: compiled cleanly, no fallback or graph-break issues** at this model size
  (`Trainer.compile_status == "enabled"`, no exception raised). `torch.compile(model)` wraps the
  model for forward/backward; checkpointing was made robust to this by always saving/loading
  through `self._raw_model` (the pre-compile module reference) rather than the compiled
  wrapper's `state_dict()`, sidestepping any version-dependent `_orig_mod` key-naming behavior.
- **Quality:** val_loss 3.4991 (ppl 33.09), **+0.0014 vs control (3.4977)** — within noise, as
  expected (compilation doesn't change the math, only how it's executed).
- **Speed:** ~535,417 tok/s — the **fastest run in the wave**, ~18% faster than the uncompiled
  control (~455,133 tok/s), even after amortizing the one-time graph-capture cost paid early in
  the 1500-step run.
- **Conclusion:** torch.compile is a real, free win on this hardware — worth defaulting to
  `compile: true` for future CUDA runs on the 5090 (M/L-tier confirmations, the phase-9 hero
  run). Still NOT recommended on the Mac/MPS path per CLAUDE.md's existing guidance, which this
  run doesn't test or contradict — it only establishes the CUDA side of the story.
