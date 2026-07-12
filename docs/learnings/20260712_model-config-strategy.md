# Sequence length vs. token count vs. model size — what does each one actually buy you?

*Discussion session 2026-07-12, following the gpuhub GPU benchmarking work (D-030/D-031/D-032).
Questions: is 512 tokens too short to see a model "perform well"? Does token count matter more
than sequence length? Do we need a "decent enough" model to see technique differences, or can
tiny models show them too? What's the minimum config for each kind of learning goal? How does
this reconcile with eventually wanting a model that can hold a short, coherent chat (~2k context)?*

## The core confusion to clear up: sequence length and token count are different axes

**Sequence length (`max_seq_len`)** is the size of the window the model can see in one forward
pass — its working memory for a single example. It's an **architectural ceiling**: a model
trained with `max_seq_len=512` cannot attend to token 600 of a document at all, full stop,
regardless of how well-trained it is. (This project's `GPT.forward()` currently enforces this as
a hard `ValueError` — see RW-5 below.)

**Token count** (how many tokens the model has been trained on, cumulatively, across the whole
run — Chinchilla's `N` in `L(N) ≈ compute`) is what actually drives **quality**: how fluent, how
knowledgeable, how low the loss. This project's own D-015 sizing already leans on this: L-tier
(105M params) targets ~2.1B tokens specifically because Chinchilla's ~20-tokens-per-parameter
rule says that's roughly where a 105M model stops being "data-starved" relative to its capacity.

**A useful analogy:** sequence length is the size of the window you're reading through at any
one moment; token count is how many books you've read in total through that window. A model
with a small window (512) that has read a huge number of books (2.1B tokens) will be a genuinely
fluent, knowledgeable reader *of short passages* — it just can't hold a whole novel's plot in
view at once. A model with a huge window (2048) that's only read a few pages (a few million
tokens) will still produce garbage, because it hasn't learned language yet — the window size
didn't help.

**So: does 512 "hurt" the model's performance?** Not in the sense of making individual sentences
less fluent or less correct — that's governed by tokens-per-param, not window size. What it DOES
limit: any task that needs more than 512 tokens of context to make sense — a multi-turn
conversation, a long document, anything requiring the model to refer back beyond ~380-400 words
(512 tokens ≈ 380-400 English words at this project's ~1.5 tok/word fertility, D-014). For raw
"is this a coherent sentence/paragraph" fluency, 512 is not a bottleneck at all. For "can this
model hold a short back-and-forth chat," it plausibly is — see the capstone section below.

**Does training at a longer sequence length teach anything token count alone can't?** A little,
yes — but it's a secondary effect, not the primary one. Long-range dependencies (a pronoun
referring back 800 tokens, a callback to something said 3 paragraphs ago) can only be *learned*
if the training sequences are long enough to contain them in the first place — no amount of
short-window training tokens will teach that skill, because the pattern literally never appears
in any single training example. This is exactly the mechanism phase 5 Wave B's positional-encoding
ablation is designed to probe (see below).

## Can tiny models actually show technique differences, or do you need "decent enough" scale first?

**Mostly yes, tiny models show real, useful signal** — and this is not a hopeful guess, it's the
literal design premise this whole project already committed to in **D-001**: *"On an M4 GPU a
100M model trains at roughly a few thousand tokens/sec; most published ablations (nanoGPT
speedruns, scaling-law papers) run the same way: sweep small, confirm big."* This matches how the
field actually operates — RMSNorm vs LayerNorm, SwiGLU vs GELU, RoPE vs learned positional
embeddings, AdamW vs Muon: these show measurable, usually *consistent-direction* effects from
~10M params all the way up to frontier scale. That's precisely why they're standard practice
today — someone validated them small first.

**But there's a real rigor step in between "ran two configs" and "trust the difference," and
phase 5 already has it built in as its mandatory first task:** a **seed-noise study** — the exact
same baseline config run 3x with different seeds, to measure how much two runs of the *identical*
config disagree just from randomness. Every later ablation verdict has to quote this noise floor
and show its effect exceeds it. This is the actual scientific answer to "is my model big/well-
trained enough to see this difference" — it's not about param count in isolation, it's about
signal vs. noise at whatever scale you're running. A 10M model can absolutely show a real,
above-noise-floor difference between RoPE and learned positional embeddings; it's just important
to have measured the noise floor first rather than assume any observed gap is real.

**What does NOT reliably transfer from tiny to large scale (be honest about the limits):**
- **Magnitude**, not direction — a technique might win by +0.05 loss at 10M and only +0.01 at
  100M (or vice versa). This is exactly why the project's plan already has M-tier "confirmation"
  runs (D-001) for the flagship findings before trusting them into the L-tier recipe.
- **MoE-style benefits** (Wave F) are muted at tiny scale almost by construction — the whole
  point of MoE is "more total capacity for the same active compute," which matters most once
  dense scaling starts hitting diminishing returns, a regime this project's models (9-105M
  params) aren't really in. The *mechanics* (routing, load balancing, the aux-loss-free trick)
  absolutely do reproduce and are worth learning at small scale — just don't expect a dramatic
  loss win from MoE itself at S-tier the way you might expect from RoPE or SwiGLU.
- **Emergent capabilities** (few-shot in-context learning, chain-of-thought) — these tend to
  appear only well above even this project's L-tier (105M). Not a concern here: this project has
  never targeted emergent reasoning, it targets a fluent, honestly-scoped small model plus a
  clear understanding of *why* each technique helps or doesn't.
- **Qualitative "feel"** — a 10M model's sample text is going to read as somewhat garbled
  regardless of which technique wins, because baseline fluency is already limited at that scale.
  The *loss number* will show the real difference even when the human-readable sample doesn't
  make it obvious. Don't judge S-tier ablations by "does this read better" — judge by val_loss
  vs. the noise floor. (This is worth remembering yourself when reading phase-5 sample outputs.)

## Minimum requirements, mapped onto the project's already-existing phase 5 plan

This project's phase 5 spec (read at session start) already answers "what's the minimum config
for each kind of learning" — it's baked into which tier each Wave is scoped at:

| Wave | Topic | Tier | Why this tier is enough |
|---|---|---|---|
| A | Norms & activations | S | Small, well-established, param-matched diffs — exactly what tiny-scale ablation is good at |
| B | Positional encodings + length-extrapolation | S (train), eval at 1024/2048 | Train cheap at 512; the whole point is testing whether RoPE/ALiBi *generalize* beyond trained length without needing expensive long-context training runs — see RW-5, this needs a code fix first |
| C | Attention variants (incl. MLA) | S + inference bench | Architecture mechanics reproduce at small scale; KV-cache byte math is exact regardless of scale |
| D | Optimizers & schedules (incl. Muon) | S, many short runs | Deliberately cheap — the value here is running *many* comparisons (6-8), which only works if each one is fast |
| E | Efficiency & memory | measurement-focused | Not a quality question at all — bf16/fp32, grad-checkpointing tradeoffs are directly measurable at any scale |
| F | MoE, MTP | S | Mechanics-focused per the caveat above — expect to learn *how* these work, not dramatic loss wins |
| G | Data & scaling (incl. domain mix, mini scaling law) | **M tier used deliberately** | This is the one Wave where the *question itself* is "how does this change with scale" — needs multiple sizes by definition |

**The pattern: S-tier (~10M params, seq_len 512, 50-100M tokens/run per `docs/EXPERIMENTS.md`,
minutes on cheap cloud GPU per D-030) is sufficient for almost everything** — this isn't a
compromise, it's the correct, field-standard design for isolating one variable at a time cheaply.
M-tier is reserved specifically for the one Wave (G) where scale *is* the variable being studied.
L-tier is reserved for the phase-9 capstone, where the goal shifts from "isolate a variable" to
"assemble the best-found recipe into one real model."

## The capstone's context-length need is a separate, deliberate decision — not an ablation setting

Your instinct that a usable chat needs ≥2k context is reasonable and worth taking seriously — a
handful of conversational turns plus a system prompt/history easily exceeds 512 tokens once
formatting overhead is included. But this is a decision about the **phase-9 L-tier capstone
specifically**, not something that should ripple back into every S-tier ablation config. Two
things reinforce that these should stay decoupled:

1. **"Chat" behavior itself mostly comes from phase 8 (SFT/LoRA/DPO), not pretraining.** Even a
   perfectly-trained L-tier base model, however long its context window, will just do raw text
   continuation until it's instruction-tuned. Context length is necessary-but-not-sufficient for
   a coherent chat model — the instruction-following behavior is a separate phase's job.
2. **RoPE (already this project's default, D-016) is specifically one of the position encodings
   known to generalize reasonably beyond its trained length** — Wave B's own ablation will give
   you real, project-specific evidence (not just literature claims) about how well RoPE
   extrapolates on *this* corpus/model before you have to commit to the L-tier's exact
   `max_seq_len`. That's the right order of operations: let Wave B inform the capstone's context
   decision, rather than guessing now.

**Logged as RW-5** (`docs/PROGRESS.md`): `GPT.forward()` currently hard-rejects any sequence
longer than `model_config.max_seq_len` with a `ValueError` — found incidentally while GPU
benchmarking seq_len scaling this session. This blocks both Wave B's extrapolation probe
(needs eval-only forward passes beyond the trained length) and, later, deliberately training the
L-tier capstone at a real ≥2048 `max_seq_len` if that's what Wave B's findings support. Not fixed
today — flagged for whoever picks up Wave B.

## The reassurance: your instinct is exactly right, and it's already this project's design

You said it yourself: "my learnings are not tied to one large good model but many different
versions." That's precisely D-001's founding rationale, written before any training happened —
tiered models specifically so ablation stays fast and cheap (S-tier), confirmation happens at
moderate cost (M-tier), and the "one good model" ambition gets exactly one dedicated shot at the
end (L-tier, phase 9) once the recipe is actually evidence-based rather than guessed. Wanting to
run dozens of cheap S-tier experiments rather than a few expensive ones isn't a compromise on
learning — as this session's GPU benchmarking found, a full S-tier ablation run now costs about
$0.02-0.03 on a rented 5090 (D-032) — it's the approach that maximizes how many "why does this
help or not" questions you actually get to answer.

## Related
- D-001 (tiered model strategy — the original rationale this discussion re-derives with real
  numbers), D-015 (tier sizing / Chinchilla budgets), D-030/D-031/D-032 (GPU capacity + cost per
  run), RW-5 (the `max_seq_len` code limitation this discussion surfaced).
- `docs/phases/phase5_ablations.md` — the wave-by-wave plan this note maps tiers onto.
- `docs/EXPERIMENTS.md` — the S-tier protocol (50-100M tokens/run, noise-floor-first rule).
