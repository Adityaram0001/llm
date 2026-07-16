# Wave F deep dive: DeepSeekMoE, aux-loss vs aux-loss-free balancing, and Multi-Token Prediction

*Discussion session, 2026-07-16, right after the Wave F implementation/run session (D-044). This
note exists so you don't have to re-read 3 notes.md files + the registry + DECISIONS.md D-044 to
remember what happened and why — everything here traces back to those artifacts (linked at the
bottom) if you want the raw numbers again. It also goes a level deeper than any single one of
those files does on its own: the mechanism math, the control-theory framing for why the two
balancing methods converge at different speeds, the full arithmetic behind the active-param
match, and a throughput angle that didn't make it into the original writeup.*

## The shape of the wave, in one paragraph

Wave F asked three questions with three ~98.3M-token runs on the RTX 5090 (~8-14 min each): does
DeepSeekMoE's "many small experts" idea actually buy real quality at matched *active* compute?
Does *how* you balance expert load (a gradient-based auxiliary loss vs DeepSeek-V3's gradient-free
bias trick) change the outcome? And does adding a cheap extra training signal (Multi-Token
Prediction) help the main next-token objective? The honest answers: **yes, clearly, for MoE's
capacity story (-0.09 val_loss, >4x the noise floor, on both balancing methods) — with a real,
substantial throughput cost the token-budget framing doesn't show; the two balancing methods tie
on final quality but differ sharply in how fast they *get there*, a clean textbook illustration of
gradient-driven vs bounded-step control; and MTP shows nothing distinguishable from noise at this
scale.** There's also a fourth thing this wave taught, arguably the most reusable lesson of the
three: a real measurement bug slipped into the first attempt at the MoE runs, and the way it was
caught (checking a delta against the noise floor before believing it, not just checking the run
"completed") is now the fifth time this exact discipline has caught a real problem in this
project.

## 1. Why this wave didn't need a new control (unlike Wave C)

Quick note before the substance, because Wave C (D-038) needed a fresh control (`n_heads=4`
instead of the project default 3, because GQA-2 is undefined at 3 heads) and Wave D (D-039)
needed one too (batch/grad_accum changed which tokens a given step sees). Wave F reused
`20260713_p5_s-wave-d-control` (val_loss 3.4977) directly — same `micro_batch=64,grad_accum=2`,
same seed, same everything except the architecture axis under test. No new control needed because
neither MoE nor MTP change anything about *how data is sampled*, only what the model computes
with it. Good hygiene check to keep making before every wave: does this wave's config change
touch anything the loader's `(seed, step)` addressing depends on? If not, the existing control is
still valid.

## 2. DeepSeekMoE, mechanism first

**The core idea in one sentence:** instead of one dense feed-forward layer that every token runs
through in full, split that same total FFN "job" into many small, independent FFNs ("experts"),
and let a small router decide, *per token*, which few of them actually get to see it. Every token
still costs about the same to process (a small number of small FFN forward passes instead of one
big one), but the *model* now has many more distinct FFN "specialists" to draw from than a dense
model of matched active compute could ever afford.

**Why fine-grained (many small) experts instead of a few big ones?** This is DeepSeekMoE's actual
contribution over earlier MoE work (Shazeer '17, Switch Transformer '21, which used fewer/larger
experts). Imagine 2 giant experts vs 8 small ones, same total FFN parameter budget either way.
With top-2 routing, "2 giant experts" means every token gets EITHER expert A OR expert B OR both
— there's no way to combine partial expertise from more than 2 distinct specialists. With 8 small
experts and top-2 routing, a token can combine any 2 of 8 = 28 possible *pairs* of specialists.
More fine-grained splitting means more combinatorial routing flexibility for the same total
capacity — this is the "fine-grained expert segmentation" DeepSeekMoE's paper title refers to.

**Why a shared expert, always on?** Every one of the 8 routed experts in our setup is competing
to specialize — but some knowledge is genuinely useful to *every* token (basic morphology,
extremely common function words, etc.), and making every routed expert re-learn that shared
baseline wastes their limited capacity on redundant common knowledge instead of differentiation.
The shared expert (always active, no routing decision, no gating weight — it just always
contributes) absorbs that common-knowledge burden so the 8 routed experts are freed up to actually
specialize.

## 3. The active-param-matching arithmetic, worked in full

This is the part of the config that's easy to gloss over but is actually the whole point of the
comparison being fair. The S-tier dense control's FFN (SwiGLU, `ffn_mult=2.6667`):

```
dense_hidden = int(ffn_mult * d_model) = int(2.6667 * 192) = 512
dense FFN params (SwiGLU = 3 matrices: gate, up, down) = 3 * d_model * hidden
                                                          = 3 * 192 * 512 = 294,912 / layer
```

Wave F's MoE config: 8 routed experts + 1 shared expert, top-2 routing. For any given token, the
"expert-equivalents" doing real work are: 1 shared (always) + 2 routed (top-2) = **3 total**. To
make the ACTIVE compute per token match the dense FFN, each individual expert's hidden dim has to
be the dense hidden divided by that count of active-equivalents:

```
expert_hidden = round(dense_hidden / (n_shared + top_k)) = round(512 / 3) = round(170.667) = 171
```

That rounding matters for interpreting the exact numbers: `171 * 3 = 513`, a tiny **+0.19%**
overshoot versus the dense control's 512 — active compute is matched to within a fraction of a
percent, not identical to the last decimal, which is exactly the kind of thing you'd want to
double-check before trusting a "matched" claim rather than taking it on faith.

**Per-expert and per-layer arithmetic:**
```
per-expert params (SwiGLU, hidden=171)   = 3 * 192 * 171 = 98,496
router params (linear, d_model -> 8)     = 192 * 8       = 1,536
9 experts (8 routed + 1 shared) + router = 9 * 98,496 + 1,536 = 887,000 / layer   (TOTAL)
active per token (1 shared + 2 routed)   = 3 * 98,496              = 295,488 / layer  (ACTIVE)
```

Times 15 layers: **13.32M total FFN params** (vs the dense control's 4.42M — **3.0x** more total
FFN capacity) but only **4.43M active FFN params/token** (vs the dense control's 4.42M — a
**0.2%** difference, matched). Whole-model totals: **18.61M vs the dense control's 9.71M** (the
gap is almost entirely the FFN; embeddings/attention/norms are untouched, identical between the
two configs — `embed 3.07M, attn 2.21M, norms 0.01M` in both).

This is the number that makes the quality comparison meaningful: **the model that won (-0.09
val_loss) was NOT allowed to spend more compute per token than the control** — it won by having
more distinct places to *store* what it learns, not by doing more work to produce each
prediction.

## 4. The routing mechanism, and how the two balancing methods actually differ

**Router → gate → dispatch, every forward pass** (`src/llmlab/model/moe.py`, `MoEFFN.forward`):

1. A plain linear layer (`d_model -> n_experts`, no bias) produces one routing *logit* per expert,
   per token.
2. `softmax` over those logits gives `gate_probs` — this is a real probability distribution over
   all 8 experts, and it's the thing that ends up DIFFERENTIABLE (gradients flow through it during
   training).
3. `top_k` (=2) picks the two highest-scoring experts *for the purpose of selection*. For
   `aux_loss` balancing, selection uses the plain logits. For `bias_free` balancing, selection
   uses `logits + routing_bias` — a per-expert additive nudge (more on this below).
4. The COMBINATION weight for each selected expert is still read from the original, unbiased
   `gate_probs` (not the biased selection score), renormalized so the two selected weights sum to
   1. This split — bias affects *who gets picked*, never *how much a picked expert counts* — is
   deliberate and is exactly DeepSeek-V3's own design choice, not an implementation detail I added
   incidentally.
5. Each of the 8 experts runs its small FFN only on the tokens routed to it (a masked
   gather/index_add, not a dense matmul over the whole batch) and the shared expert runs on
   everything unconditionally; outputs are combined by the weights from step 4.

**`aux_loss` balancing** (Switch/GShard-style): adds a term to the LOSS itself —
```
f_i = (tokens routed to expert i) / (total tokens * top_k)     # a HARD count, no gradient
P_i = mean(gate_probs[:, i])  across the batch                  # the SOFT probability, has gradient
aux_loss = n_experts * sum_i(f_i * P_i)         # summed per layer, then summed across all 15 layers
```
`f_i` is deliberately stop-gradient (it's just a count) — the thing that actually gets pushed
around by backprop is `P_i`. If expert `i` is currently overloaded (`f_i` high), this term
*directly* penalizes the router for assigning it high probability, gradient magnitude scaling with
how overloaded it currently is. That's a **proportional-control** signal: bigger imbalance -> bigger
correction, every single step, automatically.

**`bias_free` balancing** (DeepSeek-V3 S2.1.2): no loss term at all. Instead, after every
OPTIMIZER step, a per-expert bias buffer gets nudged by a FIXED size:
```
routing_bias[i] += update_rate * sign(mean_load - load[i])      # update_rate = 0.001
```
Notice this is `sign(...)`, not the imbalance magnitude — an expert that's wildly overloaded and
an expert that's only slightly overloaded both get exactly the same size correction (-0.001) per
step. This is closer to **bang-bang / fixed-step integral control**: the direction is always
right, but the step size never adapts to how far off you currently are.

## 5. The balancing-speed finding — a clean, real control-theory illustration

This table (per-expert-load `std/mean` across the 8 experts, lower = more balanced) is the wave's
cleanest result, and it's the one piece of this wave's story that isn't just in the notes.md files
verbatim — it's worth seeing the FULL trajectory, not just the two snapshots quoted there:

| step | aux_loss std/mean | bias_free std/mean | ratio (bias_free / aux_loss) |
|------|-------------------|---------------------|------|
| 10   | 0.236             | 0.406               | 1.7x |
| 200  | 0.026             | 0.166               | 6.4x |
| 400  | 0.022             | 0.102               | 4.6x |
| 600  | 0.012             | 0.075               | 6.3x |
| 800  | 0.019             | 0.023               | 1.2x |
| 1000 | 0.009             | 0.015               | 1.7x |
| 1490 | 0.011             | 0.009               | 0.8x |

Reading this as a story: both start equally imbalanced at init (makes sense — a fresh random
router has no reason to be balanced, and neither method has done anything yet). By step 200,
`aux_loss` is already tight (0.026) while `bias_free` is still 6.4x worse (0.166) — the
proportional-control signal reacted fast to the large initial imbalance, exactly as the mechanism
predicts (`P_i * f_i`, with `f_i` large for the overloaded experts right after init, produces a
correspondingly LARGE gradient early on). `bias_free`'s fixed `±0.001` nudge just can't move that
fast — it needs roughly 4x as many steps (until ~step 800) to close the same gap, because its step
size doesn't scale with how far off it started. By the very end, they're statistically
indistinguishable (0.011 vs 0.009 — if anything `bias_free` is *slightly* tighter, likely because
its fixed small step size doesn't overshoot the way a large gradient-based correction can).

**Why this matters beyond "one method is faster":** it's a genuinely reusable intuition for
reading any two balancing/regularization mechanisms — one that responds proportionally to error
will always win the RACE to a stable point, but a bounded, fixed-step corrector can still reach
the SAME final point given enough time, and comes with a real benefit the proportional method
doesn't have (zero gradient interference with the main loss — DeepSeek-V3's actual motivation,
not an incidental side-effect). If this project ever trains something where load balance needs to
be right almost immediately (e.g. a much shorter run, or expert load feeding into some other
downstream metric early in training), that's exactly when the choice between these two methods
would stop being a wash and start mattering.

## 6. The measurement bug — full postmortem, because the pattern is the real lesson

This deserves its own section because it's the fifth time in this project a "the automated run
completed and reported a number" result turned out to need independent verification before it
could be trusted (previous instances: D-022's list-aliasing bug, D-023's resume/SIGINT bugs,
D-032's incomplete GPU sweep, D-042's wandb entity silently failing every upload while printing
"done.").

**What happened, mechanically:** `GPT.forward()` computes the main next-token cross-entropy, then
(when `moe` is configured) adds `aux_loss_weight * moe_aux_loss` to get the combined training
objective — correct and necessary, since `train_step`'s `loss.backward()` needs that combined
value to actually train the router via the aux-loss signal. The bug: `Trainer.evaluate()` was
reading THAT combined value for `val_loss` too, instead of isolating pure CE. Two things made this
look like a real, plausible result instead of an obvious crash:
- `moe_aux_loss` is *summed* across all 15 MoE layers, and each layer's aux loss sits around 1.0
  at good balance (`n_experts * sum(f_i * P_i)`, which equals exactly `n_experts * (1/n_experts) =
  1` when perfectly balanced, by construction) — so the aggregate is ~15, not ~1. Multiplied by
  `aux_loss_weight=0.01`, that's **+0.15** silently added to the aux_loss run's reported val_loss.
- `bias_free`'s aux_loss is *exactly* zero by design (that balancing method has no loss term at
  all) — so its buggy number was, by coincidence, already the correct number. Nothing about that
  run looked wrong on its own.
- The two numbers together (aux_loss "3.5642", bias_free "3.4093") told a totally coherent,
  totally wrong story: "aux_loss balancing is meaningfully worse than bias_free" — a ~0.15 gap,
  10x the noise floor, easily mistaken for a genuine finding worth writing into DECISIONS.md.

**How it was actually caught:** not by re-reading the code line by line, but by the project's
standing discipline (`docs/EXPERIMENTS.md`'s noise-floor rule) — before writing any verdict, every
delta gets checked against the 0.015-0.02 D-035 noise floor, and a *within-family* comparison
(two variants of the SAME technique, same architecture, same token budget) showing a 0.15 gap
should have been surprising on its face. That "does this number make sense given everything else
I know" reflex is what caught it, not a specific test written to guard against this specific bug
(that test — `test_evaluate_val_loss_excludes_aux_terms` — was written AFTER, as the fix's
regression guard).

**The fix, and why it's shaped the way it is:** `GPT.forward()` now stores pure CE in
`self.last_aux_metrics["ce_loss"]` BEFORE any aux term gets added to the returned `loss`.
`Trainer.evaluate()` reads `ce_loss`, not `forward()`'s return value. `train_step`'s `train_loss`
metric is DELIBERATELY left unchanged (still the combined objective) — this matches an existing
precedent already in the codebase (z-loss, Wave D) where `train_loss` has always meant "whatever
the optimizer actually saw," while `val_loss` is the one number that has to stay comparable across
every wave this project has ever run or ever will run. That asymmetry — training metrics can be
whatever's useful for debugging optimization, but the one comparison metric has to be sacred and
consistent — is worth remembering the next time this project adds ANY new auxiliary loss term
(MTP's own loss follows the same rule already, which is why `mtp_loss` never touched `val_loss`
either, and why the MTP run's numbers didn't need any correction).

## 7. Multi-Token Prediction, mechanism and the index arithmetic

**The chain, one depth at a time.** The main trunk produces a hidden state `h_i` at every position
`i` (using causal context `0..i`) that predicts token `i+1` — completely standard. MTP adds one
(or more) extra sequential steps: depth 1 takes `h_i` (dropping the trunk's very last position,
since we're about to need one token beyond it) and the TRUE embedding of token `i+1` (teacher-
forced — we already know it from the training data, this isn't the model's own guess), concatenates
their RMSNorm'd versions, projects back down to `d_model`, and runs that through one more full
transformer Block. The result predicts token `i+2`, through the SAME shared `final_norm` +
`lm_head` the main path uses (no separate output vocabulary projection — deliberately, so the
extra head can't just learn a totally different representation space).

**Why this needs careful index bookkeeping** (this is the part of `GPT._mtp_loss` that's easy to
get subtly wrong): every depth's retained sequence gets ONE position shorter than the depth
before it (you need one more real future token every time you go one step deeper), and — this is
the detail worth internalizing — because the trimming always happens from the RIGHT end (`h[:,
:-1, :]`), the retained positions are ALWAYS the original sequence's left-aligned prefix
`0..(T-d-1)`. That's what makes it safe to reuse RoPE/ALiBi unmodified inside the MTP block: those
position encodings are computed fresh, starting at position 0, for whatever length tensor they're
given — and since the MTP subsequence's positions really are `0..(T-d-1)` in absolute terms (never
shifted, only shortened), "start at 0" is exactly correct, not a convenient approximation.

**Why it's train-time only, and why that's not a shortcut.** DeepSeek-V3 itself drops the MTP
modules entirely at inference — they exist purely to shape what the TRUNK learns during training
(the idea being: a trunk trained to also support "predict 2 steps ahead, given today's true next
token" might build more temporally-coherent internal representations even for its 1-step task).
That's why `generate()` never touches `mtp_heads` at all — the extra ~0.52M params and one extra
block's compute are a pure training-time cost, never paid at inference.

## 8. Why MTP's result is genuinely inconclusive, not a quiet negative

+0.0167 vs the control's noise floor of 0.015-0.02 — this sits almost exactly ON the boundary, not
comfortably inside or outside it. Two things are worth separating here, because they point in
different directions:
- The MTP head's OWN loss (predicting t+2) is real and learns cleanly: 9.71 at init (right at
  `ln(16000)=9.68`, the expected random-init baseline) down to 3.78 by the end — not degenerate,
  not stuck, genuinely a harder task than the main 1-step prediction (3.78 vs the main path's
  3.51 final CE) which makes sense: predicting 2 tokens ahead from only ONE extra token of
  lookahead is intrinsically harder than 1-token prediction, even with the true intermediate token
  handed to you.
- Whether that extra learning signal helps the MAIN task is the open question, and this run can't
  answer it cleanly — the delta is too close to the noise floor to call either way. DeepSeek-V3's
  own reported MTP benefit was measured at a vastly larger scale and token budget, where small
  per-token signal improvements have far more training steps to compound. A single depth-1 head,
  98.3M tokens, and a 10M-param trunk may simply be too small a lab for this particular technique's
  benefit to surface above the noise — which is a statement about SCALE, not about the technique
  being wrong.

## 9. The angle the original notes.md files didn't fully capture: throughput

`docs/EXPERIMENTS.md` rule 4 says judge every ablation on val loss AND wall-clock — Wave F's
writeup led with the token-budget comparison (both runs saw the same ~98.3M tokens) but didn't
put a number on the wall-clock cost, so here it is, pulled fresh from `metrics.jsonl`
(`tokens_per_sec`, median over each run's post-warmup steps):

| run | median tok/s | vs control |
|---|---|---|
| dense control | 487,858 | — |
| MoE (either balancing method) | ~223,600 | **~2.18x slower** |
| +MTP | 428,232 | ~1.14x slower |

MoE's routing is NOT free on the 5090, despite matching active FLOPs/token on paper. The likely
cause: instead of one big, GPU-efficient matmul (the dense FFN), each MoE layer does 9 SMALL
matmuls (8 routed experts' masked subsets + 1 shared expert over the full batch), each with its
own kernel-launch overhead and its own comparatively poor GPU utilization at small
per-expert batch sizes — the exact "many small operations vs one big one" launch-overhead story
Wave E already established for micro-batch/grad-accum factorization (D-040), showing up again
here in a different guise. MTP's ~14% slowdown is much more modest and matches expectations for
"one extra Block's forward+backward per step," no routing overhead involved.

**What this changes about the phase-9 recommendation:** MoE's -0.09 val_loss win is real at fixed
TOKENS, but at fixed WALL-CLOCK (train for the same amount of GPU time instead of the same number
of tokens), the dense control would see roughly 2.18x more tokens in that time — and Wave E
already showed more tokens matters a lot at this scale. Whether MoE still wins at fixed compute-
time is an open question this wave didn't answer; it would need an equal-wall-clock rerun (dense
control trained ~2.18x longer, or MoE trained ~2.18x fewer steps) to settle. This throughput angle
wasn't computed during the original Wave F session, so D-044 doesn't mention it — added to
PROGRESS.md's Parking lot from this discussion session instead of retroactively editing D-044.

## 10. The recurring theme: total capacity vs. active compute, across three waves now

Wave F's MoE result isn't an isolated finding — it's the third time this project has now measured
something in the "more capacity, same active compute" family, and it's worth seeing them side by
side because they're subtly different versions of the same idea:

| wave | what grew | is the extra capacity ACTIVE on every token? | result |
|---|---|---|---|
| Wave E, weight tying off (D-040) | +3.07M (untied unembedding matrix) | **Yes** — every forward pass uses the full unembedding matmul | -0.0278 (real, but not a fair "does tying cost quality" test — this is just "more active compute" in disguise) |
| Wave C, MLA (D-038) | *shrinks* the KV cache, params roughly flat | n/a — this was a memory/latency story, not a capacity one | quality flat, cache 3.2x smaller |
| Wave F, MoE (D-044) | +8.90M (13.32M total FFN vs 4.42M dense) | **No** — only ~1/3 of the routed capacity (3 of 9 expert-equivalents) is active per token | -0.09 (real, genuinely a capacity-without-compute story) |

The Wave E comparison is the important contrast: untying the embedding ALSO added total
parameters and ALSO improved quality, but every one of those extra parameters is used on every
single token (the unembedding matmul touches the whole matrix regardless of which token it is) —
so that result is really just "more active compute helps," unsurprising, and explicitly flagged in
D-040 as NOT a clean tying-vs-quality test for exactly this reason. MoE's result is the
qualitatively different, more interesting claim: the extra 8.9M FFN parameters are mostly IDLE for
any given token (a token only ever touches 3 of the 9 expert-equivalents), yet the model still
gets measurably better — because different tokens get to use DIFFERENT slices of that extra
capacity. That's the actual value proposition of sparse MoE that dense scaling can't replicate:
decoupling "how much the model knows in total" from "how much compute any single prediction
costs." Worth having this distinction crisp if the phase-9 capstone ever needs to choose between
"just make the dense model bigger" and "add MoE routing" as two ways to spend a parameter budget —
they are NOT interchangeable levers, even when they produce similar-looking val_loss deltas.

## 11. What actually changes in the phase-9 recipe

- **DeepSeekMoE is a strong capstone candidate IF the L-tier's total-parameter/memory budget can
  absorb ~2x FFN growth** — the quality win is real and substantial at matched active compute, not
  a fluke of one run (both balancing methods independently confirm it).
- **Balancing method choice is a non-issue for final quality** — pick `bias_free` if the
  DeepSeek-V3 rationale (zero gradient interference with the main loss) matters philosophically or
  at a scale where that interference might start to bite; pick `aux_loss` if fast, predictable
  balancing early in training matters more (e.g. if some downstream measurement depends on
  balanced routing being reached quickly).
- **MoE's real cost is wall-clock, not quality** — budget ~2.2x the GPU-hours of an equivalent
  dense run before committing to it for a real (expensive) capstone run; this wasn't visible from
  the token-budget-only framing in the original notes.md files.
- **MTP is not yet earning its keep at this scale** — not disqualified, just unproven; would need
  either a bigger model/token budget or a `loss_weight`/`n_predict_tokens` sweep before it's worth
  including in a real recipe.
- **Process lesson, not a recipe lesson:** any FUTURE auxiliary loss term added to `GPT.forward`
  must follow the `ce_loss`-isolation pattern from day one (store pure CE before adding anything
  else to the returned loss) — this bug class is now closed for this project structurally, not
  just fixed for this one instance.

## Links

- Decision log: `docs/DECISIONS.md` D-044 (full design rationale, the bug postmortem, options
  considered)
- Per-wave summary: `docs/results/ablation_log.md`, "Wave F" section
- Figure: `docs/results/wave_f_deepseek_specials.png` (`scripts/plot_wave_f.py`)
- Registry rows: `experiments/registry.csv`, `20260716_p5_s-wave-f-{moe-auxloss,moe-biasfree,mtp}`
- Per-run hypothesis/observation/conclusion: `experiments/20260716_p5_s-wave-f-*/notes.md`
- New code: `src/llmlab/model/moe.py` (`MoEFFN`), `src/llmlab/model/mtp.py` (`MTPHead`), wired
  through `src/llmlab/model/{block,gpt}.py` and `src/llmlab/train/trainer.py`
  (`update_moe_bias` hook, `last_aux_metrics`/`ce_loss` isolation)
- Related prior findings this note connects to: D-040 (Wave E weight-tying, the "is this just
  more active compute" question), D-038 (Wave C MLA, the params-vs-cache-vs-compute distinction),
  D-035 (the noise floor this wave's bug detection leaned on)
- Papers: Dai et al. '24 (DeepSeekMoE), DeepSeek-AI '24 (DeepSeek-V3, S2.1.2 aux-loss-free
  balancing + S2.2 Multi-Token Prediction), Shazeer et al. '17 (original sparse MoE), Gloeckle et
  al. '24 (Better & Faster LLMs via Multi-token Prediction, the non-DeepSeek MTP lineage)
