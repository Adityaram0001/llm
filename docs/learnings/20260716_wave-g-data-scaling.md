# Wave G deep dive: domain-mix specialization, the overfitting gap, and why bigger models overfit a fixed pool faster

*Discussion session, 2026-07-16, right after the Wave G implementation/run session (D-045). This
note exists so you don't have to re-read 11 notes.md files + the registry + DECISIONS.md D-045
to remember what happened and why — everything here traces back to those artifacts (linked at
the bottom). It also goes a level deeper: the mixing-weight sampling mechanism, the full epoch/
budget arithmetic behind every config choice, a two-sided domain-val-loss measurement that
wasn't part of the original run (computed live in this session from the saved checkpoints), and
the throughline connecting all three studies to one underlying idea.*

## The shape of the wave, in one paragraph

Wave G asked three data/scaling questions across 11 runs on the RTX 5090 (~30 min total
wall-clock): if you mix a specialized domain into training, how much does it cost the general
model, and does it actually buy anything back? What does classic overfitting look like when you
force a small model to see the same small pool of text many times? And how does loss scale with
parameter count at a fixed token budget? The honest answers: **domain mixing is a clean, one-way
trade — general quality degrades monotonically with domain share, and (verified in this session,
not in the original run) domain quality improves monotonically in almost exact proportion, so
there's no free lunch but there IS a real, working knob;** **the train/val gap opens exactly as
predicted well before val loss itself would ever show a problem, making the gap the earlier and
more honest overfitting signal;** and **bigger models overfit a small, repeated token pool faster,
not slower — the single most important empirical fact for planning the L-tier capstone's data
budget**, discovered specifically by comparing each run's best (early-stopped) value against its
final (end-of-budget) value rather than trusting the final number alone.

## 1. RW-4's domain corpus: why curation quality mattered more than book count

RW-4 had been open since phase 1 — "the user wants a finance/wisdom-flavored model" was on record
but no actual books existed. This session pulled from a local (outside-the-repo) pre-scraped
Gutenberg catalog, category-tagged. The catalog's categorization is auto-generated from PG's own
metadata and is **noisy in a specific, predictable way**: it tags books by loose keyword/subject
overlap, not genre, so "Finance" pulled in *The Merchant of Venice* (four separate editions),
Mark Twain's *The Gilded Age*, and dozens of novels merely ABOUT money rather than teaching
anything about it. "Self Help" was even worse — 1,677 entries, the overwhelming majority being
Horatio Alger's juvenile rags-to-riches novels (86 books alone, one author), which are thematically
adjacent but stylistically miles from the register a "finance/wisdom" flavor is supposed to mean.

**Why this mattered enough to hand-vet rather than bulk-import:** the training stream is going to
literally learn this text's *prose register*, not just its topic tags. Mixing in 80 juvenile
adventure novels because a keyword-matcher called them "Self Help" would have taught the model a
children's-story voice, not a finance/wisdom voice — actively working against the stated goal
while looking, by book count, like exactly what was asked for. So the actual work was: filter to
categories that plausibly hold real nonfiction (Finance, Investing, Economics, Business, Self
Help, Personal Development, Wisdom & Philosophy), then hand-select ~60 titles by recognizable
author/genre (Adam Smith, Ricardo, Bagehot, Bastiat, Keynes, Veblen for economics/finance theory;
Samuel Smiles, James Allen, Orison Swett Marden, Russell Conwell, Elbert Hubbard, P.T. Barnum,
Ralph Waldo Trine, W.D. Wattles for the "New Thought"/practical-success genre; Ford, Taylor,
Tarbell, Gilbreth for business/management), landing on **62 books**. Two candidates (Adam Smith's
*Wealth of Nations* and *Theory of Moral Sentiments*) turned out to already be in the general
112-book philosophy corpus — caught by cross-checking Gutenberg IDs before adding, not after.

**A cheap, general litmus test worth keeping for any future corpus expansion:** does the
candidate list read, in aggregate, like the *register* you're trying to teach, not just the
*topic*? Keyword-matched categories answer "is this about X" — they don't answer "does this
sound like X," which is the thing that actually ends up in the model's weights.

## 2. Domain-mix ablation: the mixing mechanism, the budget arithmetic, and the two-sided result

**Mechanism.** `MixedSourceLoader` (built back in phase 4, with RW-4 already in mind — see its
own docstring) samples a source index per training example from a categorical distribution over
sources, weighted by each `Source.weight` normalized to sum to 1. For `sources: [books_dict
weight=0.75, domain_books weight=0.25]`, that means: for every training example in a batch,
independently roll a weighted coin — 75% chance it's a random window from the general 17.66M-token
books+dictionary pool, 25% chance it's a random window from the 6.76M-token domain pool. Over a
whole batch of 64 sequences this converges to *roughly* a 75/25 split by sequence count (not
exactly — it's a stochastic draw per example, not a deterministic partition), which is exactly
enough precision for an ablation like this.

**Why the budget had to shrink from ~98.3M tokens to 49.15M.** The domain pool is small — 6.76M
raw tokens. The phase-5 spec's own design rule for this ablation is "domain repetition ≤4
epochs" (to avoid the result being contaminated by outright memorization of a tiny pool rather
than genuine domain adaptation). Do the arithmetic at the standard budget: 50% of 98.3M tokens =
49.1M domain tokens ÷ 6.76M pool = **7.3 epochs** — already blowing the ≤4 rule for exactly the
data point (50% share) that most needs to be trustworthy. Halving the total budget to 49.15M
tokens brings the worst case back in bounds: 50% × 49.15M = 24.6M domain tokens ÷ 6.76M = **3.6
epochs**. This is the same "does the design constraint survive the most extreme test point"
check that showed up in Wave D's WSD-fork budget math and Wave F's active-param arithmetic — a
recurring habit worth keeping: before locking in a shared budget across several settings of one
knob, check the arithmetic at the *most extreme* setting, not the median one.

**One consequence of this choice, worth flagging honestly:** these 4 runs' val_loss numbers
(3.98–4.14) are **not comparable to the D-035 noise floor** (0.015–0.02), which was measured at
the standard ~98.3M-token/1500-step budget. They're only valid compared *against each other*,
all four sharing this wave's own budget and seed. This is a real, if narrow, tradeoff of
shrinking the budget to keep the repetition rule intact — the fix would be growing the domain
corpus (more raw tokens → more room at the standard budget → directly comparable numbers), which
is exactly what `docs/results/recipe.md` flags as an open question before L-tier.

**The result — a strictly monotonic, one-way cost to general quality:**

| domain share | general val_loss | Δ vs 0% | domain val_loss* | Δ vs 0% |
|---:|---:|---:|---:|---:|
| 0% | 3.9800 | — | 4.6938 | — |
| 10% | 4.0153 | +0.0353 | 4.6383 | −0.0555 |
| 25% | 4.0549 | +0.0749 | 4.5806 | −0.1132 |
| 50% | 4.1442 | +0.1642 | 4.5355 | −0.1583 |

*Domain val_loss was NOT logged during training (only general val was, by design — a domain
probe is phase-6 work). It was computed in this discussion session by loading each run's
`ckpt/best.pt` and running `Trainer.evaluate()`'s exact CE-loss computation against
`domain_books_val.bin` (the 3 held-out domain books) — a quick, code-unchanged analysis pass,
not a new pipeline feature.

**Why this table is the actual finding, not just the general-val column from the original run:**
on its own, "general val loss gets worse as you add domain data" could describe either (a) a
real specialization trade, or (b) the model just getting generically worse/more confused by a
messier mixed stream, with no upside at all. The domain-val column rules out (b) decisively —
domain-val loss improves monotonically and by almost the same absolute magnitude the general-val
loss worsens (0.164 general cost vs 0.158 domain gain, from 0% to 50%). This is a textbook
specialization/generality tradeoff with both sides actually measured, not assumed. A qualitative
check of the generated samples confirms it's not just a number: at 700 steps (~46M tokens), the
0%-share run's "ephemeral (adjective):" completion produces generic, slightly garbled prose
("Exalalal", "_Ponand_"), while the 50%-share run's completion for the same prompt reads
"Export of the State... an increased number of individuals... in large countries... in 1862"
— recognizably finance/economics register, not philosophy-register noise.

**Practical takeaway:** there is no "free" domain share — every point costs general quality
and buys domain quality in roughly equal measure. The choice of 10–25% (recipe.md's
recommendation) isn't "the point where the cost stops," it's "the point where the trade still
favors keeping the model broadly useful while giving it real finance/wisdom flavor" — a
judgment call, not a discovered threshold, since this sweep shows no plateau by 50%.

## 3. Multi-epoch overfitting lab: why the gap, not the val level, is the tell

**Setup.** Books-only pool (no dictionary — its short, structured entries would otherwise
confound a pure "does the model start memorizing prose" study), 14,141,233 tokens, trained for
exactly 1/4/16 "epochs" — meaning `max_steps` was chosen so that `max_steps × tokens_per_step`
equals 1×/4×/16× the pool size: 216/864/3456 steps at 65,536 tokens/step. Evaluated against the
matching `books_only_val.bin` split (2 held-out books) rather than the general val set, so
train and val are drawn from the exact same distribution — the only variable is how many times
the model has seen the training half of that distribution.

**Result:**

| epochs | train_loss | val_loss | gap (val − train) |
|---:|---:|---:|---:|
| 1 | 5.427 | 5.695 | +0.268 |
| 4 | 4.104 | 4.448 | +0.344 |
| 16 | 3.207 | 4.128 | +0.921 |

**The gap opens exactly as the phase-5 spec predicted — but notice what does NOT happen: val
loss never gets worse.** It improves a lot from 1→4 epochs (5.695→4.448) and then only a little
more from 4→16 (4.448→4.128), essentially flattening out, while train loss keeps falling the
whole time, all the way to 3.207. This is the textbook first stage of overfitting — the model
has enough capacity to keep squeezing the train loss down by memorizing specifics of the 14.14M
token pool (exact phrasing, rare word sequences, which book a given sentence came from), but that
memorization stops transferring to the held-out books once the "free," genuinely-general
improvements are used up. **The gap is the earlier, more sensitive signal** — if you only
watched val loss level, 16 epochs would look like a mild win over 4 epochs (4.128 vs 4.448); only
the widening gap (+0.921 vs +0.344) tells you that "win" is increasingly built on memorization,
not generalization, and that this trend will not reverse for free with more of the same data.

**Why val loss doesn't turn around and rise here, unlike the scaling law study below:** this is
the S-tier model (9.71M params) at 512 seq_len — the same size used everywhere else. It simply
doesn't have enough capacity, at this data volume, to fully memorize a 14.14M-token pool badly
enough to actively hurt generalization within only 16 epochs. Section 4 shows what happens once
you *do* have enough relative capacity.

## 4. The mini scaling law: the fit is secondary, the overfitting-by-size finding is the point

**Framing.** Kaplan (2020) and Chinchilla (2022) both fit `L(N) = a·N^-α + c` — loss as a smooth
power-law function of parameter count N, with irreducible loss `c` as N→∞. Their studies used
enormous, effectively-fresh (non-repeated) token streams. This is a deliberately "mini" version:
4 points (5M/10M/25M/50M params), fixed 200M-token budget, over the *same* 17.66M-token
books+dictionary pool as every other S-tier ablation — meaning **every point in this study sees
the identical ~11.3 epochs of repetition**, so param count is the only thing that changes. `lr`
was held fixed at 1e-3 across all 4 sizes — a real, flagged simplification; a rigorous study
would retune lr per size (muP-style), since larger models often want a smaller lr or more warmup
to realize their full capacity.

**Sizing the 4 points.** `head_dim=64` is fixed project-wide (D-016), so `n_heads = d_model / 64`
must be an integer, and tied embeddings + `vocab_size=16000` are also fixed — leaving `d_model`
and `n_layers` as the only free knobs. A small parameter-count search across `d_model ∈
{64,96,...,640}` (multiples of 64) and `n_layers ∈ [2, 40]` found the closest exact matches:

| target N | d_model | n_layers | n_heads | actual N | error |
|---:|---:|---:|---:|---:|---:|
| 5M | 128 | 15 | 2 | 4,999,168 | 0.0% |
| 10M | 192 | 15 | 3 | 9,713,472 | 2.9% (reused `model_s.yaml` directly) |
| 25M | 320 | 16 | 5 | 24,786,240 | 0.9% |
| 50M | 384 | 25 | 6 | 50,400,384 | 0.8% |

The 10M point deliberately reuses the project's own S-tier shape rather than the search's
technically-closer alternatives — it's the most battle-tested architecture in the whole project,
and reusing it means this study's "10M" point is directly the same model every other wave has
been ablating against.

**The headline result — best (early-stopped) vs final (end-of-budget) val_loss:**

| N | best val_loss | @ step (of 3050) | @ tokens | final val_loss | overfit gap |
|---:|---:|---:|---:|---:|---:|
| 5.00M | 3.4031 | 3050 (last) | 196.7M | 3.4031 | +0.0000 |
| 9.71M | 3.2663 | 3050 (last) | 196.7M | 3.2663 | +0.0000 |
| 24.79M | 3.1655 | 2400 | 157.4M | 3.1789 | +0.0134 |
| 50.40M | **3.1701** | **1650** | 108.2M | 3.2789 | **+0.1088** |

5M and 10M are still improving at the very last logged step — they simply don't have the
capacity to fully exploit 11.3 epochs of an 17.66M-token pool within this budget. 25M starts
turning over a bit past its 2/3 point. **50M peaks barely past HALF its budget (step 1650 of
3050, ~6.1 of the ~11.3 epochs available) and then gets steadily WORSE for the rest of the run**
— by the final step, its val_loss (3.2789) is worse than 25M's final val_loss (3.1789), even
though 50M's train_loss at that point (2.285) is the lowest of any model tested, comfortably
below 25M's final train_loss (2.738). The model is unambiguously still learning *something* —
just not something that transfers to held-out text anymore.

**Why this is exactly the project's own "Muennighoff ceiling" idea, one level more concrete.**
RW-1 already flagged, back in phase 4's data planning, that the Muennighoff et al. (2023)
data-constrained scaling literature finds returns from repeating a fixed token pool diminish past
roughly ~4 epochs — that's WHY the project targeted enough fresh tokens (TinyStories + a
FineWeb-Edu sample) to keep repetition under that ceiling for the real training runs. This wave
shows the *mechanism* behind that ceiling isn't a fixed number of epochs at all — **it's a
function of how much capacity the model has relative to the pool.** A 5M-param model can chew
through 11.3 epochs of 17.66M tokens without any sign of trouble; a 50.4M-param model runs out of
genuinely new things to learn from that same pool around epoch 6 and starts actively
memorizing instead. The practical consequence for L-tier (~100M params, 2x bigger again than the
largest model tested here): **the repetition budget that was safe for phase 4/5's ablations is
very unlikely to still be safe at L-tier's parameter count** — L-tier needs either genuinely
fresh tokens or a meaningfully lower epoch count than this project's smaller models have used.

**Why the fit uses best, not final, values — and why that distinction matters beyond this one
wave.** Fitting `L(N) = a·N^-α + c` on the FINAL numbers would put the 50M point (3.2789) above
the 25M point (3.1789) — a fitted curve would either have to bend backward (violating the
power law's monotonic-decrease assumption) or simply report a worse fit while hiding the real
reason: at that specific N, the number measures "how overfit is this model by the end of a
budget it outgrew," not "how good can this model get." Fitting on the BEST value instead
(3.1701 for 50M) restores monotonicity and reports the number each model's *capacity* actually
earned. This is the same underlying discipline as D-044's aux_loss/val_loss bug and D-042's
wandb-entity bug: a plausible-looking automated number (here, "final step's logged value") isn't
automatically the right number to report — always sanity-check what a metric is actually
measuring before trusting it as *the* answer.

**The fitted curve itself, read honestly:** `L(N) = 11909.67 · N^-0.694 + 3.102` — alpha ≈ 0.69,
much steeper (and much noisier) than Chinchilla's own ~0.34. This is expected, not a red flag:
4 points spanning only one order of magnitude, at a fixed (non-muP) learning rate, in a
data-constrained (not fresh-token) regime, will not recover a textbook-precision exponent. Treat
this fit as a qualitative "returns diminish with scale, and diminish faster than a naive read of
the literature would suggest at THIS project's specific data budget" signal — not something to
extrapolate numerically out to L-tier's ~100M params.

## 5. The throughline connecting all three studies

Every section here is really one question wearing three different costumes: **given a fixed,
small amount of real text, what happens as you ask more of a model — more repetition, more
domain specialization, more parameters?** The consistent answer across all three: **something
has to give, and it shows up as a widening gap between what the model has memorized and what it
can generalize**, whether that gap is measured as train-vs-val (section 3), domain-vs-general
val (section 2), or best-vs-final val at increasing N (section 4). None of these are new facts
in the abstract — they're all textbook — but seeing all three appear from the SAME
17.66M/14.14M/6.76M-token pools, in the same session, with the same S-tier architecture, is a
much more concrete, load-bearing intuition than reading about any one of them in isolation.

This also directly connects to two open threads elsewhere in the project: Wave E already showed
that quality-neutral engineering choices (bf16, `torch.compile`, batch/accum factorization) can
still have large real costs (memory, wall-clock) that a pure val_loss comparison hides; Wave F's
parking-lot item (MoE's untested equal-wall-clock comparison) is the same "which axis are you
actually holding fixed, and does that choice quietly favor one option" question this wave's
best-vs-final distinction is built entirely around. Any future wave that runs multiple settings
of one knob to a shared, hard-coded step count should ask, before trusting the final numbers:
*is every setting equally well-served by stopping at exactly this step, or does the choice of
stopping point itself favor some settings over others?*

## What this means for phase 9 (see `docs/results/recipe.md` for the full consolidated table)

- **Domain share:** budget 10–25% for the finance/wisdom flavor — there is no discovered
  "free" amount, so this is a deliberate choice of how much general quality you're willing to
  spend, not a threshold this data revealed.
- **Repetition budget:** do NOT assume the ~11.3-epoch repetition used throughout phase 5's
  S-tier ablations is safe at L-tier's ~100M params. This wave's own scaling-law finding says the
  opposite — bigger models overfit a fixed pool faster. Plan on genuinely fresh tokens (or a
  meaningfully lower epoch count) for the hero run.
- **Growing the domain corpus:** since domain share and repetition tolerance both push toward
  "more raw domain tokens" being the actual lever worth pulling before L-tier, not just "should
  we use 15% or 20%."

## Related files

- Registry rows: `20260716_p5_s-wave-g-domainmix-{00,10,25,50}`, `20260716_p5_s-wave-g-epochs-
  {01,04,16}`, `20260716_p5_scaling-{5m,10m,25m,50m}` in `experiments/registry.csv` — each has
  its own `notes.md`.
- `docs/DECISIONS.md` D-045 (the full decision entry).
- `docs/results/ablation_log.md`'s Wave G section (5-line summary).
- `docs/results/recipe.md` (consolidates this wave + all others for phase 9).
- `docs/results/wave_g_data_scaling.png` / `notebooks/07_scaling_law.ipynb` (the figures + code
  behind every number in this note, including the domain-val two-sided table computed live in
  this session).
- `docs/DECISIONS.md` D-015/D-020 (the original Muennighoff-ceiling framing this wave's scaling
  law makes concrete).
