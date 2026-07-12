# 20260712_p5_s-wave-b-nope — Wave B run 4/4: RoPE -> NoPE (no explicit positional signal)

**Hypothesis:** NoPE relies entirely on the causal mask for order information (some research
suggests decoder-only causal attention can implicitly recover a notion of position even without
explicit encoding) — expected worse than RoPE at trained length, and untested going in whether
extrapolation would degrade gracefully or catastrophically.

**Observation:** final val_loss 3.6997 (ppl 40.43) vs baseline's 3.5037 — delta +0.196, real and
consistent (worse than RoPE, similar magnitude to `learned`'s +0.227). Length-extrapolation probe
(512/1024/2048): val_loss 3.6997 -> 4.2074 -> 6.5957 (ppl 40.43 -> 67.18 -> **731.91**) — a
catastrophic collapse, by far the sharpest degradation of any encoding tested (RoPE: 33->37->46
ppl; ALiBi: 33->32->32 ppl, actually improving).

**Conclusion:** NoPE trains adequately at the length it was trained at (implicit position
information from the causal mask alone is enough for *this* task at trained length) but has
**zero mechanism for handling unfamiliar lengths** — confirms that whatever weak positional
signal NoPE learns is tied tightly to the specific trained context window, not a generalizable
notion of relative position the way RoPE/ALiBi provide. The sharpest contrast in the whole probe:
NoPE (worst extrapolation) vs ALiBi (best, actually improves) sit at opposite ends despite both
being config-only-different swaps of the same otherwise-identical model.
