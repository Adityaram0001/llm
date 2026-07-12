# 20260712_p5_s-wave-b-alibi — Wave B run 3/4: RoPE -> ALiBi

**Hypothesis:** ALiBi (Press et al. '21) should match or slightly trail RoPE at the trained
length (both are relative, in-attention mechanisms) but should extrapolate at least as gracefully
— ALiBi's whole selling point in the paper is *better* length extrapolation than any alternative
tested at the time, including RoPE.

**Observation:** final val_loss 3.4830 (ppl 32.56) vs baseline's 3.5037 — delta -0.0207, just
past the 0.015 noise floor (D-035) and consistent from early training onward (not just a late
lucky read) — ALiBi is marginally **better** than RoPE at the trained length here, not just
competitive. Length-extrapolation probe (512/1024/2048): val_loss **improves** with more context
— 3.4830 -> 3.4683 -> 3.4552 (ppl 32.56 -> 32.08 -> 31.67) — compare to RoPE's same probe on
`20260711_p4_s-baseline`, which degrades (3.5037 -> 3.6052 -> 3.8216, ppl 33.24 -> 36.79 -> 45.68).

**Conclusion:** the standout result of Wave B and the clearest reproduction of a paper's claim in
this project so far. ALiBi's additive recency bias means more context is unambiguously more
signal (nothing to "forget how to use" past the trained length), while RoPE's rotation-based
relative encoding, though it degrades gracefully, still degrades — this small-scale test
reproduces the ALiBi paper's headline finding almost exactly. Recommend ALiBi as a strong
candidate if a future phase (M/L-tier, or the phase-9 capstone's chat-context goal, RW-5) needs
robust long-context behavior without training natively at that length.
