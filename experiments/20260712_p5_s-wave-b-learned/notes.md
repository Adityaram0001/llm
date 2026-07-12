# 20260712_p5_s-wave-b-learned — Wave B run 1/4: RoPE -> learned absolute position embeddings

**Hypothesis:** learned absolute position embeddings (GPT-1/GPT-2 style) should train fine at the
trained length but cannot generalize past it (fixed-size lookup table, one row per position up
to `max_seq_len`) — the classic RoPE/ALiBi selling point is exactly this limitation.

**Observation:** final val_loss 3.7311 (ppl 41.73) vs baseline's 3.5037 (ppl 33.24), delta +0.227
— real and consistent, RoPE trains noticeably faster/better at this scale. Length-extrapolation
probe (`scripts/eval_extrapolation.py`): seq_len=512 gives 3.7311 (matches training); seq_len=1024
raises `ValueError: sequence length 1024 exceeds max_seq_len 512` — the expected, by-design
failure (RW-5's fix keeps this guard specifically for `learned`/`sinusoidal`, since their tables
are physically sized to `max_seq_len`).

**Conclusion:** confirms both halves of the hypothesis. RoPE's relative-position mechanism is a
better inductive bias than absolute position lookup at this scale, AND learned embeddings are
structurally incapable of the length-extrapolation probe entirely — not a training artifact, a
hard architectural ceiling.
