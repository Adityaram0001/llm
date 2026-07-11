# Phase 6 — Evaluation suite

**Goal:** one command that scores any checkpoint on a fixed battery; understand what each
metric does and doesn't tell you. Built once, used by phases 5/8/9 forever after.
**Effort:** 1–2 sessions. Can start right after phase 4.

## Deliverables

1. **`src/llmlab/eval/`** + **`scripts/evaluate.py`** →
   `python scripts/evaluate.py --ckpt experiments/<run>/ckpt/best.pt [--suite core]`
   writes `eval_results.json` into the run folder and (optionally) a registry column update.
2. **Core suite:**
   - **Val perplexity** on our held-out books & dictionary val split (separately! domain ppl
     differs — that's informative).
   - **Bits-per-byte** on the same data (tokenizer-independent — explain why this exists).
   - **Dictionary probes** (our special sauce, uses `dictionary.jsonl` val entries):
     (a) *definition completion ppl*: "‹word› (‹pos›): …" scored on the gold definition;
     (b) *multiple-choice define*: right definition vs 3 shuffled wrong ones by total
     log-likelihood → accuracy (chance=25%);
     (c) *cloze*: mask the headword given the definition.
   - **Domain probes — finance/wisdom (RW-4):** same MC-by-loglik pattern as dictionary
     probes, built from held-out domain text: finance-term definitions, proverb/maxim
     completion, "sound advice vs nonsense" pairs (data factory can generate these). Measures
     whether the domain mixing actually steered the model.
   - **Generation battery**: 15 fixed prompts (story openers, "Define X:", book-style prose,
     + finance/wisdom prompts like "The first rule of saving money is") at temp 0.8/top-p
     0.95, saved side-by-side across checkpoints for eyeballing.
   - **Repetition/diversity metrics**: distinct-n, repetition rate on generations.
3. **Standard benchmarks (tiny-model-appropriate, via manual implementation not lm-eval-harness
   — implementing them IS the lesson):**
   - HellaSwag (log-likelihood MC, expect near-chance — discuss why),
   - LAMBADA-style last-word accuracy on held-out book passages (homemade version).
4. **`notebooks/08_eval_deep_dive.ipynb`**: run suite on 3 checkpoints of the baseline run
   (early/mid/final) → watch capabilities emerge; calibration plot (predicted prob vs actual);
   discussion of benchmark contamination and why our val books must stay out of training.

## Decision points
- Fixed eval battery contents FROZEN after this phase (changing evals mid-project invalidates
  comparisons — log this as a decision).
- Sampling params for the generation battery.

## Learning checkpoints
- Why ppl comparisons require identical tokenizer+data; what bits-per-byte fixes.
- Why MC-by-loglik works for base models that can't follow instructions.
- What "emergence" looks like vs smooth metric improvement at our scale.

## Exit criteria
`evaluate.py --suite core` runs on best baseline checkpoint in <10 min; results JSON schema
stable; M3 milestone; PROGRESS/DECISIONS updated.
