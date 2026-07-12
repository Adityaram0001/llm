# 20260711_p4_s-baseline

**Hypothesis:** THE S-tier reference run for phase 4/5 comparisons. lr=1e-3 was carried
forward from the overnight lr sweep's control run rather than re-derived -- reviewed 2026-07-12
and ratified as D-021's original default (see D-025): the sweep showed 1e-3 strictly ahead of
0.3x/3x at every checkpoint, so this is not actually a deviation from D-021, just a confirmation.

**Observation:** 1500 steps, 98.3M tokens, wall-clock 2h25m (~11,000-11,400 tok/s, matching
D-022's calibration). val_loss: 9.55 (step 0, ~ln(16000)=9.68, near-random) -> 5.22 (step 100,
the initial cliff) -> 4.18 (step 400) -> 3.655 (step 900) -> **3.504 (step 1400, best)** --
textbook power-law flattening, biggest gains earliest, diminishing but still-real gains late.
Perplexity 33.2. grad_norm stable throughout, no spikes. Samples (`samples/step_*.txt`): step 0
is pure noise; by step 800 the model produces fluent grammar AND has picked up the corpus's
Socratic-dialogue register specifically (`"Soc. As to the question, I must now explain..."`) --
a legible fingerprint of D-011's philosophy-heavy book selection, not generic English. By step
1400, longer and more structurally confident prose, still semantically wandering as expected at
9.71M params. Note: the `"ephemeral (adjective):"` dictionary-format prompt, which produced
`**Word** (pos.):` style output in the earlier `p4_smoke` run, drifts toward book-prose instead
by the later baseline checkpoints -- plausibly because dictionary entries are a small minority
of the S-tier corpus relative to books; worth revisiting when phase 6 builds the
dictionary-completion eval probe.

**Conclusion:** first genuinely successful S-tier pretrain. Confirms the training engine
(loader, trainer, lr schedule, checkpointing) works end to end on real data at real scale, not
just on the smoke run's 150 steps. This run is the baseline every phase-5 S-tier ablation will
diff against (`baseline_run` in future registry rows). lr=1e-3 is ratified, not provisional.
