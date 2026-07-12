# 20260712_p5_s-wave-a-postnorm — Wave A run 2/4: pre-norm -> post-norm

**Hypothesis:** post-norm (normalize after adding the residual, original Transformer/GPT-1 style)
should be markedly less stable than pre-norm at this depth (15 layers) without extra warmup care
— a deliberate negative-result control per the phase-5 spec, not expected to be competitive.

**Observation:** val_loss finished at 6.8810 (ppl 973.6) vs baseline's 3.5037 — catastrophically
worse, but **not via the "instability" most people picture (loss spikes/NaN)**: grad_norm never
exceeded 1.52 across the whole run (grad_clip=1.0 mostly held it in check), and train_loss
dropped fast to ~6.8 by step ~150 then flatlined for the remaining 1350 steps. Generated samples
at step 1400 are degenerate punctuation/fragment soup ("not his,, that,, of in of,. the to, of
by all;,,..."), confirming the model got stuck, not just numerically unstable.

**Conclusion:** post-norm's failure mode here is **early stagnation, not blow-up** — the
un-normalized residual stream makes gradients from later layers too diluted/noisy to keep
improving past a shallow local optimum, exactly the mechanism Xiong et al. '20 describe for why
pre-norm enables deep transformers to train without careful warmup schedules. Confirms pre-norm
as the right default; would need a much more careful (likely much longer, lower-lr) warmup to
give post-norm a fair fight, which is out of scope for this ablation's fixed-compute comparison.
