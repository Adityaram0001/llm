# Phase 6 eval suite deep dive: how each metric actually works, what it does and doesn't tell you, and a real bug found while checking

*Discussion session, 2026-07-17, right after the phase 6 implementation session (D-046). This
note exists so you don't have to re-read `notes.md` + the registry + `docs/DECISIONS.md` D-046/
D-047 + six source files to remember what the eval suite does, why it's built the way it is, and
— this is the part a "what happened" summary can't give you — exactly what it does and does not
measure. Writing this note is also what surfaced a real, 100%-reproducing bug in one of the
suite's own metrics (D-047), because the discipline of re-deriving every claim against the real
tokenizer instead of trusting the implementation session's own docstrings is what this project's
whole track record (D-022/23/32/42/43/44/45) is built on. That bug is documented in full below,
not just cross-referenced — this note is meant to stand on its own.*

## 1. The shape of phase 6, in one paragraph

Phase 6 built one thing: a fixed, frozen battery that scores any checkpoint on five angles —
corpus-level perplexity (split by data type), three dictionary probes exploiting the corpus's
unique GCIDE-dictionary asset, domain probes for RW-4's finance/wisdom steering question, a
15-prompt generation battery with diversity metrics, and two standard benchmarks implemented by
hand (HellaSwag, a homemade LAMBADA-style task). All five reduce, underneath, to one shared
primitive: "how likely does the model think this specific continuation is, given this specific
prompt" (`src/llmlab/eval/scoring.py`). Understanding that one primitive well is most of what's
needed to understand the whole suite — sections 2-4 below go deep on it, including a real bug
that a careless version of it would produce (and, as it turns out, the shipped version does).

## 2. The shared primitive: log-likelihood scoring, and why length has to be normalized

A base model — no instruction tuning, no chat format — has no notion of "pick option A/B/C/D."
What it *does* have, for any text you hand it, is a probability: `P(continuation | prompt)`,
computed by teacher-forcing the continuation's tokens through the model and multiplying (in log
space, summing) each token's predicted probability of being the token that actually comes next.
Comparing that number across several candidate continuations — and picking the highest — is
instruction-free by construction. This is the entire trick behind MC-by-loglik evaluation
(`mc_by_loglik` in `scoring.py`), used by the dictionary probes, domain probes, and HellaSwag.

**Why the raw SUM of log-probabilities is the wrong thing to compare.** Every token's
log-probability is negative (probabilities are ≤1, so `log(p) ≤ 0`). Summing N negative numbers
is monotonically related to N — all else equal, a SHORTER continuation has fewer negative terms
to add and so, almost by construction, a higher (less negative) sum, regardless of whether it's
actually the better answer. Worked illustration (toy numbers, not from a real run — the point is
the mechanism):

| continuation | tokens | per-token logprob | sum logprob | mean logprob |
|---|---|---|---|---|
| "a small furry mammal that barks" (correct, long) | 7 | -1.1, -0.9, -1.3, -0.8, -1.0, -1.2, -0.9 | **-7.2** | **-1.029** |
| "a rock" (wrong, short) | 2 | -2.5, -2.0 | **-4.5** | **-2.25** |

Judging by raw sum, the WRONG short answer wins (-4.5 > -7.2) even though every individual token
of the wrong answer is far less likely than every token of the right one (-2.5/-2.0 vs
~-0.8 to -1.3). Judging by mean (dividing by token count first), the correct answer wins cleanly
(-1.029 > -2.25) — this is exactly why `score_continuation` returns both `sum_logprob` and
`mean_logprob`, and why `mc_by_loglik` always ranks candidates by the MEAN, never the sum. This
is the same convention GPT-3's paper uses for its own MC-by-loglik evaluations (Brown et al.
'20, appendix G) — not something invented for this project, a known necessary correction.

## 3. The prompt/continuation boundary trick — and the real example that foreshadows section 4

BPE tokenization is not word-aligned — the SAME text can tokenize differently depending on what
comes immediately before it, because merge rules can pull a leading space into the following
word (or not) depending on context. Concretely, checked against the real `hf_bpe_16k` tokenizer:

```
tokenizer.encode("excessive desire for wealth")
  -> ['excess', 'ive', ' desire', ' for', ' wealth']       (5 tokens — standalone)

tokenizer.encode("avarice (noun): excessive desire for wealth")
  -> [...'):' , ' excessive', ' desire', ' for', ' wealth']  (the same phrase, but "excessive"
                                                               is now ONE token, ' excessive',
                                                               because it follows a space)
```

If you score a continuation by encoding the prompt and the continuation SEPARATELY and
concatenating the token IDs, you get a token sequence the model never actually learned to expect
in that exact form — it saw `" excessive"` (one token) constantly during training, essentially
never `"excess"` immediately followed by `"ive"` as two freshly-started tokens. `scoring.py`'s
`encode_prompt_continuation` exists specifically to avoid this: encode the WHOLE
`prompt + continuation` string ONCE, then split — the standard lm-eval-harness-style technique.

**The subtlety that makes this only a partial fix**, and the direct setup for section 4: the
implementation splits by counting `len(tokenizer.encode(prompt_text).ids)` and slicing the
jointly-encoded sequence at that many tokens. This assumes the separately-encoded prompt is an
exact token-for-token PREFIX of the joint encoding. It usually is. It is not always — and the
`"avarice (noun): "` example above is exactly a case where it fails, because the prompt's
trailing space gets absorbed into `" excessive"` in the joint encoding, but the separately-encoded
prompt (asked to tokenize `"avarice (noun): "` all on its own, with no continuation to inform it)
has no reason to leave that space un-merged — it just emits a bare, standalone space token.

## 4. The bug this note's own fact-checking found (D-047, RW-6 — not fixed yet)

Here is exactly what goes wrong, step by step, using the real tokenizer:

```
prompt_text       = "avarice (noun): "
continuation_text = "excessive desire for wealth"

tokenizer.encode(prompt_text).ids                        -> 8 tokens, decodes to "avarice (noun): "
tokenizer.encode(prompt_text + continuation_text).ids    -> 11 tokens, decodes correctly

encode_prompt_continuation() returns:
  prompt_ids       = the 8-token standalone prompt encoding
  continuation_ids = joint_ids[8:]   # slice at "8", the standalone prompt's own length

joint_ids[8:] turns out to be [' desire', ' for', ' wealth']  -- only 3 tokens.
"excessive" (joint_ids[7], the merged ' excessive' token) is silently left OUT of both halves.
```

`score_continuation` then builds `full = prompt_ids + continuation_ids` (8 + 3 = 11 tokens — the
right COUNT, wrong CONTENT) and feeds it to the model. Decoded, that sequence reads:

> `"avarice (noun):  desire for wealth"` — a double space, and **the word "excessive" is gone.**

This is not a rare edge case. I checked it against all 3,281 entries in the real
`data/clean/val/dictionary.jsonl`, exhaustively, not a sample:

| probe | boundary shape | entries checked | corrupted |
|---|---|---|---|
| **dictionary_probes (a) definition-completion** | prompt ends `": "` (bare trailing space), continuation starts with a word | 3,281 | **3,281 (100%)** |
| dictionary_probes (c) cloze | prompt ends `":"` (no space), continuation supplies its own `" word"` | 3,281 | 0 (0%) |
| domain_probes (all 24 items × 4 choices) | prompt ends in punctuation/word, continuation supplies `" " + choice` | 96 | 0 (0%) |
| benchmarks.run_hellaswag | same safe shape as domain_probes | 2,000 (500 real rows × 4 endings) | 0 (0%) |
| benchmarks.run_lambada_style | same safe shape by construction | not exhaustively swept — structurally identical to the two verified-safe rows above | — |

**The mechanical reason it's 100%, not occasional:** `dictionary_probes.py`'s prompt template is
`f"{word} ({pos}): "` — ALWAYS ends in a bare trailing space, by construction, for every one of
the 3,281 entries. And in a 16,000-vocab BPE trained on English prose, the overwhelming majority
of English words have a learned `" word"` merged token (that's what a BPE tokenizer's most
common merges look like) — so the "prompt ends in space, continuation starts with a mergeable
word" collision isn't bad luck, it's close to guaranteed by the template's own shape. Every other
probe in the suite happens to put the "risky" boundary space on the CONTINUATION's own side
(`" " + choice`, `" " + word`) instead of the prompt's trailing side — which is exactly why
they're all safe: a continuation that already starts with an explicit space character can't have
that space silently absorbed into a "prompt-side" token, because the space was never claimed by
the separately-encoded prompt in the first place.

**What this invalidates, precisely:** `dictionary_probes.definition_completion_ppl` — every
number quoted for it (baseline: 101.28; milestone steps 150/750/1500: 440.62 / 123.35 / 98.23;
and the corresponding curve/line in `notebooks/08_eval_deep_dive.ipynb`'s first figure) is a real
number computed on real model outputs, but scored against systematically corrupted target text.
**What survives:** every other number in every `eval_results.json` written this phase — corpus
ppl/bpb, MC accuracy, cloze ppl/accuracy, domain probes, HellaSwag, LAMBADA-style, generation
diversity — none of them touch this code path.

**A second, milder issue found in the same audit, in the MC probe.** `dictionary_probes.py`'s
multiple-choice sub-probe doesn't call `encode_prompt_continuation` at all — it encodes the
prompt and each of the 4 choice texts independently. This does NOT corrupt content (concatenated
and decoded, the text round-trips correctly: `"avarice (noun): excessive desire for wealth"`,
exactly right) — it just uses a different, out-of-context tokenization for each choice (a
spot-check of 500 real dictionary entries found 51, ~10.2%, get a different token count than the
in-context form would produce). This adds noise to `mc_accuracy` without invalidating it the way
the content-loss bug invalidates `definition_completion_ppl`.

**The fix, for whenever RW-6 gets picked up** (not applied this session — discussion sessions
change no code, CLAUDE.md): stop trusting the token-count-based prefix assumption. The
`tokenizers` library exposes character offsets per token —
`tokenizer.encode(text).offsets -> [(start_char, end_char), ...]` against the ORIGINAL string.
Encode `prompt_text + continuation_text` once, then split by CHARACTER position instead of token
count: every token with `end_offset <= len(prompt_text)` is "prompt," every token with
`end_offset > len(prompt_text)` is "continuation." Verified this reconstructs the correct split
on the `"avarice (noun): "` example (the boundary-straddling `" excessive"` token, spanning
characters 15-25 against a 16-character prompt, correctly lands entirely in "continuation" since
its end offset, 25, exceeds 16). No second `tokenizer.encode(prompt_text)` call needed at all.

**Why the implementation session's own test didn't catch this**
(`test_encode_prompt_continuation_splits_at_the_boundary`, `tests/test_eval.py`): its one example
was `"The cat sat on the"` + `" mat."` — where the continuation ALREADY carries its own leading
space, the safe pattern, not the pattern `dictionary_probes.py` actually uses. **Reusable
lesson**: a shared helper's unit test should be adversarial against its REAL call sites'
boundary shapes, not just a convenient example that happens to work.

## 5. Perplexity vs. bits-per-byte, worked with this session's real numbers

From `experiments/20260711_p4_s-baseline/eval_results.json` (the phase-4 baseline's `best.pt`):

| split | ppl | bits/byte | tokens | bytes | bytes/token |
|---|---|---|---|---|---|
| books | 68.457 | 1.588 | 80,384 | 308,691 | **3.840** |
| dictionary | 18.562 | 1.301 | 98,816 | 320,117 | **3.240** |

The books/dictionary GAP looks huge under ppl (68.5 vs 18.6 — books look **3.7x** harder) but
shrinks a lot under bpb (1.588 vs 1.301 — only **1.22x**). The reconciling fact is the bytes/token
column: dictionary text tokenizes MORE densely (3.24 bytes/token vs books' 3.84) because
dictionary prose is templated and repetitive (`"**Word** (pos.): definition."`, lots of common
grammatical/definitional phrasing) — the BPE tokenizer, trained on the whole corpus, learned
longer merged tokens for dictionary-style patterns specifically. A model can be "fewer bits
of surprise per raw CHARACTER of dictionary text" (which is the fairer comparison — bpb) while
still looking dramatically "easier per TOKEN" (ppl) partly because dictionary tokens are just
denser units in the first place, not purely because the content itself is 3.7x more predictable.
The exact arithmetic connecting them: `bits_per_byte = log2(ppl) / bytes_per_token` —
`log2(68.457)/3.840 = 6.098/3.840 = 1.588` ✓, `log2(18.562)/3.240 = 4.214/3.240 = 1.301` ✓ (both
verified against the real numbers above, not approximated).

**Why this matters beyond this one comparison:** it's the concrete answer to the phase-6 spec's
own learning-checkpoint question, "why ppl comparisons require identical tokenizer+data." Any
time two things being compared were tokenized differently — different vocab size (the "v2
scale-up" parking-lot idea), or, as shown here, just different TEXT STYLES under the SAME
tokenizer — ppl alone conflates "genuinely harder to predict" with "happens to tokenize into more
pieces." bpb is the metric that survives that conflation.

## 6. Two different perplexity ESTIMATORS now exist in this codebase — and they don't quite agree, on purpose

This is a subtle methodological point worth having explicit, because it's easy to assume "val
loss" means one specific, unambiguous computation across the whole project. It doesn't, exactly:

- **The training loop's `Trainer.evaluate()`** (used for every `val_loss` in every
  `metrics.jsonl`/registry row since phase 4, including D-035's noise floor) samples a FIXED set
  of `eval_batches × eval_batch_size` RANDOM windows once at `Trainer.__init__` time
  (`fixed_eval_batches`, `src/llmlab/data/loader.py`) and reuses that exact same subsample for
  every eval call during a run. For the baseline's config (`eval_batches=32,
  eval_batch_size=16`, `seq_len=512`), that's 262,144 token-predictions sampled — MORE than the
  full 179,655-token val set, meaning windows overlap / repeat by construction.
- **Phase 6's `perplexity.evaluate_split`** does a deterministic, EXHAUSTIVE, non-overlapping
  sweep of every window in the target `.bin` file — no sampling, no repeats (beyond one token of
  intentional overlap between consecutive windows, needed so window `i`'s last input token can
  also serve as window `i+1`'s first target — see the module's own docstring).

Reconstructing one from the other, using this session's real numbers: combining the SEPARATE
`books_only_val.bin` (68.457 ppl, 80,384 tokens) and `dictionary_only_val.bin` (18.562 ppl,
98,816 tokens) perplexities by token-weighted mean NLL gives a combined estimate of
**ppl=33.333** (mean NLL 3.5066) — compared to the ORIGINAL training-loop `val_loss=3.5037`
(ppl=33.238) recorded for this exact same `best.pt` checkpoint back in phase 4. The two
estimators agree to within **0.0028 nats** (a ppl difference of about 0.1) — reassuringly close,
confirming both are measuring the same underlying thing — but they are NOT literally the same
computation, and a future session comparing a `metrics.jsonl` `val_loss` against an
`eval_results.json` `perplexity.books`/`.dictionary` number should expect them to be close, not
bit-identical, for this reason.

## 7. Statistical significance — the gap this note's own numbers expose most clearly

**This is the single most important "what's not covered" finding, and it corrects something the
phase-6 implementation session's own console output got wrong by implication.** When
`scripts/evaluate.py` printed `mc_acc 0.265 (chance 0.25)` for the baseline's dictionary MC
probe, the framing (mine, in that session) was "a real, if weak, above-chance signal." That
framing does not survive a proper significance check.

For a binomial proportion at `n` trials, chance `p=0.25`, the standard error is
`sqrt(p(1-p)/n)`. Computed for every accuracy metric this phase actually reports:

| probe | n | SE | 95% CI half-width | chance range (95%) | this session's observed value(s) |
|---|---|---|---|---|---|
| dictionary MC | 200 | 0.0306 | 0.060 | **[0.190, 0.310]** | 0.265 (baseline), 0.245/0.275/0.270 (milestones) — ALL inside |
| HellaSwag | 200 | 0.0306 | 0.060 | **[0.190, 0.310]** | 0.215 — inside |
| LAMBADA-style | 150 | 0.0354 | 0.069 | **[0.181, 0.319]** | n/a (last-word EXACT MATCH, not MC — see §8) |
| domain probes, overall | 24 | 0.0884 | 0.173 | **[0.077, 0.423]** | 0.250 (baseline, mid, final) — inside; 0.500 (early/step150) — **borderline, z=2.83** |
| domain probes, per category | 8 | 0.1531 | 0.300 | **[-0.050, 0.550]** (i.e. anything 0-55% is "noise") | see §8 |

**None of this phase's dictionary-MC, HellaSwag, or (3 of 4) domain-probe numbers are
statistically distinguishable from pure chance at the sample sizes actually used.** The correct
reading of `mc_acc=0.265` is "consistent with chance, not proof of a real above-chance signal
one way or the other" — not the more confident "weak positive signal" framing used in the
moment. This project already has exactly the right instinct for this (D-035's noise floor exists
for precisely this reason, for val_loss) — it just hasn't been extended to these small discrete
accuracy probes yet. **Concretely not built**: no formal per-probe noise floor (the D-035
equivalent for accuracy metrics), and `eval_results.json` doesn't report a confidence interval or
standard error alongside any accuracy number — only the point estimate. Both are natural, cheap
follow-ups (binomial CIs cost nothing extra to compute) flagged in the parking lot below.

## 8. A case study in why MC-by-loglik numbers need this scrutiny: the step-150 proverb result

The eval_deep_dive notebook's own capability-trajectory table shows `domain_mc_acc=0.500` at
step 150 (the EARLIEST, least-trained checkpoint) — higher than at step 750 or step 1500, which
both sit at exactly 0.250. Breaking that down by category (`eval_results_step_000150.json`):
`finance_term_accuracy=0.375`, **`proverb_accuracy=1.000` (8/8)**, `advice_accuracy=0.125`.

**8 out of 8 correct is a striking number to get from a model that's barely past its training
warmup.** Under a pure-chance null (`p=0.25`), the probability of 8/8 by luck alone is
`0.25^8 ≈ 0.0000153` (about 1 in 65,000) — technically "surprising" if you treated the proverb
category alone as a single pre-registered hypothesis test. It is almost certainly NOT that rare
in reality, for two compounding reasons worth having explicit:

1. **This wasn't a pre-registered single test.** This session (like the eval_deep_dive notebook)
   looked at 3 categories × 3 checkpoints × 1 overall = up to a dozen numbers before noticing this
   one. Checking many numbers and reporting the most extreme one without correcting for how many
   you looked at (the multiple-comparisons problem) makes "surprising-looking" results far more
   likely to appear somewhere by pure chance than the raw p-value for any single cell suggests.
2. **A more interesting, unconfirmed hypothesis worth naming rather than ignoring: MC-by-loglik
   can be gamed by a candidate's own unconditional word frequency, not real task competence.**
   The 8 proverb items' correct answers are all common, short, high-frequency English words
   ("money," "wise," "themselves," "earned," ...). If the model's early, undertrained log-
   likelihoods are dominated by a generic unigram-frequency prior (common words score higher
   almost regardless of context, because that's true of English text in general, not because the
   model has "understood" the specific proverb), that alone could produce above-chance accuracy
   on a set where the correct answer happens to skew toward more frequent words than the
   distractors — with ZERO real proverb comprehension behind it. **This is a real, known
   critique of MC-by-loglik evaluation in the broader literature** (part of GPT-3's own paper,
   Brown et al. '20, motivates a "PMI-style" normalization — dividing each candidate's
   conditional log-likelihood by its UNCONDITIONAL log-likelihood, i.e. how likely that same text
   is with NO context at all — specifically to strip out this base-rate effect). **This eval
   suite does not implement that normalization anywhere** — every MC-by-loglik probe in this
   project (dictionary, domain, HellaSwag) uses raw conditional log-likelihood only. I have not
   directly measured whether frequency bias is actually what happened at step 150 (that would
   need a follow-up: score each proverb's 4 choices with NO prompt context at all, and check
   whether the "frequency-only" ranking already matches the "with-context" ranking) — flagging it
   as a plausible, not confirmed, mechanism, in keeping with this project's discipline of not
   declaring a mechanism proven without checking it directly.

## 9. Calibration and Expected Calibration Error — going one level deeper than the notebook does

The notebook computes a **reliability diagram**: bucket every next-token prediction by the
model's own top-1 confidence (the softmax probability assigned to its single most-likely token),
then for each bucket ask "of the predictions made at roughly this confidence, what fraction were
actually correct?" A perfectly calibrated model's points sit exactly on the diagonal. **Expected
Calibration Error (ECE)** is the single-number summary: the bucket-count-weighted average gap
between predicted confidence and actual accuracy — `sum(bucket_size * |mean_confidence -
empirical_accuracy|) / total_predictions`. This session's real number, computed live over
`books_only_val.bin`'s 80,384 next-token predictions on the FINAL checkpoint:
**ECE = 0.0164** (top-1 accuracy 0.279, mean confidence 0.295) — well-calibrated in the sense
that both numbers sit close together, even though both are individually low.

**Why cross-entropy training tends to produce reasonable calibration "for free," and when that
stops being true.** Cross-entropy is a *proper scoring rule* — it is minimized in expectation
exactly when the predicted distribution matches the TRUE data-generating distribution, not by
any other strategy (a model can't get a lower expected loss by being systematically over- or
under-confident; the unique loss-minimizing strategy is honesty). This is WHY a plain
next-token-prediction model doesn't usually need dedicated calibration machinery (temperature
scaling, Platt scaling, etc.) the way a model trained with a non-proper objective, or one
evaluated far outside its training distribution, often does. **The "for free" part depends on an
assumption**: the eval data has to be reasonably IN-distribution relative to what the model
learned from. This note's own ECE number is only evidence that calibration holds ON
`books_only_val.bin` (in-distribution, held-out books) — it says nothing about whether the same
model is equally well-calibrated on genuinely out-of-distribution text (HellaSwag's web-scraped
wikiHow style, or even the dictionary/domain corpora, which differ stylistically from the books
val split). **Not measured, not covered**: calibration on any split other than books val, at any
checkpoint other than the final one, even though the code (`perplexity.evaluate_split`'s
sibling logic) would support extending this trivially.

## 10. What this eval suite covers — and, in matching detail, what it does not

This is the section you asked for directly — the honest inventory, not a marketing pass.

### Covered

- **Perplexity AND bits-per-byte, split by data type** (books vs dictionary, never conflated) —
  §5 above shows exactly why splitting AND having the tokenizer-independent metric both matter.
- **Three distinct angles on the corpus's unique dictionary asset**: forward (define-given-word,
  §4's caveat notwithstanding), discriminative (multiple-choice), and reversed (cloze,
  word-given-definition) — three different failure modes get three different chances to show up.
- **A domain-steering measurement isolated from general-quality cost** — Wave G (D-045) already
  measured "how much does mixing in domain data cost general val_loss"; phase 6's domain probes
  measure the OTHER side of that trade: "did it actually buy anything." This baseline (0% domain
  share) scoring exactly at chance on them is the correct, meaningful null result confirming the
  probes work as a discriminator (a model that saw zero domain tokens SHOULD score at chance).
- **A real, external, contamination-free sanity check (HellaSwag)** — establishes a "this is what
  near-zero real-world commonsense competence looks like at this scale" reference point future,
  bigger runs can be compared against.
- **A homemade long-range task grounded in the project's OWN held-out prose** (LAMBADA-style),
  rather than only relying on external benchmarks whose text register has nothing to do with this
  corpus.
- **Generation quality/diversity beyond loss** (distinct-1/2, seq-rep-4) — catches degenerate
  "stuck repeating itself" failure modes a pure perplexity number can hide entirely.
- **Determinism and speed**: every probe is seeded, the whole suite runs in under a minute on an
  M4 Mac (`~38s` measured, spec required `<10min`) — cheap enough to run on every future
  checkpoint as a matter of routine, not something to ration.
- **A frozen, versioned contract** (`eval_results.json`'s schema) — future checkpoints (phase 8
  fine-tunes, phase 9's capstone) are directly comparable to phase 6's own numbers without
  re-deriving what "comparable" even means.
- **`--max-examples` override** for a fast smoke-test mode, and a clean separation between "what
  the model KNOWS" (`evaluate.py`) and "how FAST/cheap it is to run" (kept deliberately in
  separate, pre-existing tooling — `scripts/bench_inference.py` from Wave C,
  `scripts/bench_activation_memory.py` from Wave E — not duplicated here).

### NOT covered (some of these are fine as-is for a base-model lab; flagged so nobody assumes otherwise)

- **No statistical significance / confidence intervals reported anywhere in `eval_results.json`**
  — §7's entire finding. The single biggest gap uncovered this session. A cheap fix (binomial CIs
  cost nothing to compute) not yet implemented.
- **No formal per-probe noise floor**, unlike D-035's val_loss noise floor — nothing establishes
  how much a probe's OWN accuracy naturally varies run-to-run (different seed, different training
  run, same architecture) the way three actual seed-controlled runs established it for val_loss.
- **No frequency/PMI-domain-conditional normalization** for any MC-by-loglik probe (§8) — a real,
  literature-documented gap, not a hypothetical one.
- **No instruction-following or chat evaluation whatsoever.** Every probe in this suite assumes a
  BASE model (that's why MC-by-loglik exists at all, §2) — there is no SFT/chat model yet (that's
  phase 8), and this suite has no mechanism to evaluate one once it exists. A chat-eval suite
  (instruction-following accuracy, a small MT-Bench-style rubric, or similar) is a wholly
  separate, not-yet-designed piece of future work.
- **No safety, toxicity, or bias evaluation.** Not a live concern given the tightly-curated,
  public-domain, 19th-century-book-plus-dictionary corpus — but worth NAMING as an absence, not
  silently assuming it was considered and dismissed, especially before any phase 8/9 model is
  used more broadly.
- **No long-context evaluation baked into the frozen suite.** The length-extrapolation work (RoPE
  vs ALiBi vs NoPE past `max_seq_len`, Wave B/D-037) lives in a separate one-off script
  (`scripts/eval_extrapolation.py`), not in `evaluate.py`. If the L-tier capstone changes context
  length or positional encoding (RW-5's still-open part b), that probe needs to be re-run by
  hand — it will not happen automatically as part of the "core" suite.
- **No throughput/latency/memory measurement in `evaluate.py`** — deliberate, not an oversight:
  `scripts/bench_inference.py` (Wave C) and `scripts/bench_activation_memory.py` (Wave E) already
  own that separately, and `evaluate.py` is scoped to "what does the model know," not "how fast."
- **No built-in cross-checkpoint statistical comparison** — the notebook does this manually
  (a pandas DataFrame + matplotlib), there's no "is checkpoint B significantly better than
  checkpoint A" helper function anywhere in `src/llmlab/eval/`.
- **No programmatic decontamination check.** HellaSwag's disjointness from a 19th-century
  book+dictionary corpus is trusted BY CONSTRUCTION (there is essentially zero chance of overlap
  given the source material), not verified by an actual n-gram-overlap scan between the corpus
  and the benchmark. Fine at this project's current scale/corpus; would stop being fine if a much
  broader web-scraped corpus (the "v2 scale-up" parking-lot idea) were ever adopted.
  Val-book/dictionary exclusion from training IS verified by construction (whole-document
  held-out split, D-012) — see the notebook's own contamination-discussion section for that half.
- **No robustness/paraphrase-invariance testing.** Every MC question is asked with exactly ONE
  fixed prompt phrasing and ONE fixed (seeded) choice-ordering shuffle — no averaging over
  multiple choice orderings or prompt templates, even though MC-by-loglik evaluations are known
  in the broader literature to be sensitive to answer order in ways unrelated to real knowledge.
- **No check that the suite itself has enough statistical power to detect a real difference when
  one exists** — phase 5 validated this indirectly for val_loss (the noise floor IS that check),
  but nothing analogous exists yet for the discrete accuracy probes. A natural test: run the
  suite against two checkpoints already KNOWN to differ meaningfully (e.g. Wave A's QK-norm
  win, -0.062 val_loss, more than 4x the noise floor) and see whether any accuracy probe actually
  moves in response — not done yet.
- **Heavy, repeated reuse of the SAME held-out val split** across every phase-4/5/6 experiment
  this project has ever run (dozens of ablations, all judged against `data/clean/val/`). The
  MODEL never trains on it (verified, §"contamination" in the notebook) — but the PROJECT itself
  has effectively used this one fixed val set as a de facto tuning signal across every decision
  made so far. This is a mild, real form of researcher/project-level "overfitting to the val
  set" over time (a known general methodology concern for any long-running project with one
  fixed benchmark) — worth naming, not something with an obvious cheap fix (a genuine held-out
  TEST set, queried only once at the very end, would be the standard mitigation, and isn't
  something this project has set up).
- **No multi-seed variance measurement for the PROBES themselves** — only one eval seed (0) has
  ever been used for which dictionary entries/HellaSwag examples get subsampled, or which
  generation-battery random draws happen. Different seeds would shift every reported accuracy
  number somewhat; that spread has never been measured.
- **Calibration is measured on exactly one checkpoint (final) and one data source (books val)**
  — §9's own caveat. Not extended to other checkpoints, other splits, or genuinely OOD data.
- **The `--suite` CLI flag has exactly one real value (`core`)** — it's a placeholder for a
  future smaller "quick" or bigger "full" suite, not yet differentiated content.
- **And, found while writing this very note: `definition_completion_ppl` is currently WRONG**
  (§4) — not a "not covered" gap, an active correctness bug in something that WAS supposedly
  covered. Listed here too so this section is a complete, honest map of the suite's current
  reliability, not just its scope.

## 11. Misconceptions corrected this session

- **"`mc_acc=0.265` (dictionary MC) is a weak positive signal."** Corrected in §7: at `n=200`,
  the 95% chance range is `[0.190, 0.310]` — 0.265 sits comfortably inside it. The correct
  reading is "not distinguishable from chance," not "weak but real."
- **"Higher ppl always means the split is genuinely harder to predict."** Corrected in §5: part
  of the books-vs-dictionary ppl gap (3.7x) is really just dictionary text tokenizing into denser
  tokens (3.24 vs 3.84 bytes/token), not a 3.7x difference in how "surprising" the underlying
  characters are — bpb (1.22x gap) is the fairer read.
- **"A striking accuracy number (like 8/8 on proverbs) at an early checkpoint is a real, if
  small, capability."** Complicated in §8: could be a real (if premature) signal, could be
  multiple-comparisons luck, could be an artifact of MC-by-loglik's known sensitivity to answer
  base-rate frequency — genuinely ambiguous with the tooling this suite currently has, and it's
  more honest to say so than to pick the most flattering interpretation.
- **Implicit assumption, corrected by direct testing rather than ever being stated out loud:**
  "`encode_prompt_continuation`'s encode-whole-then-slice trick is safe as long as SOME version
  of the standard technique is used." False in general (§3-4) — the specific token-count-based
  slicing implementation here has a real, common failure mode; the technique needs to split by
  CHARACTER offset, not token count, to be safe in general.

## 12. What actually changes for phase 7 onward

- **Do not quote `definition_completion_ppl` from any existing `eval_results.json`** until RW-6
  is fixed (D-047). Every other number in this phase's output is trustworthy.
- **Report confidence intervals (or at least the sample size) alongside every accuracy number**
  going forward, in any write-up (notes.md, DECISIONS.md, learnings notes) that cites one of
  these probes — §7's table is reusable as-is for the current n's (200/24/8/150).
- **RW-6's fix is well-specified enough to apply directly** (§4's offset-based approach) whenever
  `src/llmlab/eval/` is next touched — likely alongside phase 8's fine-tuning work, since that's
  the next phase expected to reuse this suite on a new kind of checkpoint (an SFT model).
- **The Wave G dictionary-ablation follow-up (now unblocked by these probes) should wait for
  RW-6's fix** if it plans to use `definition_completion_ppl` as its headline metric — the OTHER
  two dictionary probes (MC, cloze) are unaffected and usable today.
- **Phase 9's capstone**, when it eventually re-runs this suite on a much bigger model, is exactly
  the moment §7's "no noise floor for accuracy probes" gap would be worth finally closing — a
  bigger, better-trained model is far more likely to produce accuracy deltas big enough to need a
  real significance threshold to interpret correctly, the same way D-035's noise floor became
  necessary the moment phase 5's ablations started producing close-together val_loss numbers.

## Links

- Decision log: `docs/DECISIONS.md` D-046 (the full phase-6 build), D-047 (this session's bug
  finding, fuller technical detail than section 4 above if needed)
- Rework queue: `PROGRESS.md` RW-6 (the not-yet-applied fix)
- Phase spec: `docs/phases/phase6_evaluation.md`
- Code: `src/llmlab/eval/{scoring,perplexity,dictionary_probes,domain_probes,generation,
  benchmarks,report}.py`, `scripts/evaluate.py`
- Notebook: `notebooks/08_eval_deep_dive.ipynb` (the capability-trajectory and calibration
  figures referenced in §§7-9; note its "dict def-completion" curve is affected by D-047)
- Runs: `experiments/20260711_p4_s-baseline/eval_results.json`,
  `experiments/20260717_p6_s-p6-baseline-milestones/eval_results_step_{000150,000750,001500}.json`
- Prior noise-floor precedent this note extends to accuracy metrics: D-035 (val_loss seed-noise
  study, `docs/EXPERIMENTS.md`)
- Papers: Brown et al. '20 (GPT-3 — MC-by-loglik + length/domain normalization conventions),
  Zellers et al. '19 (HellaSwag), Guo et al. '17 (On Calibration of Modern Neural Networks — the
  ECE/reliability-diagram formalism used in §9)
