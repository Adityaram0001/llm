# 20260712_p5_s-wave-b-sinusoidal — Wave B run 2/4: RoPE -> fixed sinusoidal position embeddings

**Hypothesis:** sinusoidal (Vaswani et al. '17, no learned parameters) should behave similarly to
learned absolute embeddings — same "additive at the input, absolute position" mechanism, just
without the extra ~98K trainable parameters — expected a small or negligible difference from the
`learned` ablation.

**Observation:** final val_loss 4.9896 (ppl 146.87) vs baseline's 3.5037 — delta +1.486, by far
the worst result of Wave B, and notably worse than `learned`'s +0.227 too (val_loss 4.99 vs
3.73). The gap over `learned` is present from early training (step 300: sinusoidal 5.79 vs
learned's 4.80) and never closes. Also fails the length-extrapolation probe past seq_len=512 by
construction (fixed pe table), same as `learned`.

**Conclusion:** a genuine surprise — losing the ~98K learnable position parameters costs far more
than expected at this scale. Plausible explanation: with only `d_model=192` and no learned
adaptation, the fixed sin/cos pattern is a harder signal for the (small, 15-layer) model to fold
usefully into its representations than either a fully learned table (which can shape itself to
whatever the model finds useful) or RoPE (which acts directly inside attention, not diluted by
being summed into the token embedding before any processing). Worth flagging for Wave G's scaling
study: does this gap shrink or persist at M/L tier?
