# Parameter allocation — why "100M params" is an incomplete spec

*Discussion session 2026-07-11, after phase 2 (tokenizer = 16k BPE, D-014). Triggered by the
realization that efficient models allocate parameters differently than older/weaker ones.*

## The core insight: two kinds of parameters

A decoder-only LM's parameters split into:

1. **Embedding/unembedding** — the `vocab × d_model` lookup table (and its transpose as the
   output head). For any given token, only ONE row (d_model numbers) participates. These
   params store surface knowledge ("which vector is token 8123") but do **no computation**.
2. **Block ("active") parameters** — attention + FFN weights. EVERY one of them multiplies
   into EVERY token at EVERY layer. This is where the FLOPs happen (≈ 6 × N_active FLOPs per
   token for fwd+bwd) and where "reasoning" capacity lives.

Hence the quote that started this: a model whose param count is dominated by embeddings is
closer to a lookup table than a thinking machine. This is also why scaling-law papers
(Kaplan '20, Chinchilla '22) count **non-embedding** params: that's what predicts loss.

## Our actual numbers (vocab = 16,384, D-014)

Embedding table = 16,384 × d_model. **Weight tying** (Press & Wolf '16) shares it with the
output head — without tying you pay twice.

| Config | Embed (tied) | Total ≈ | Embed share | Active params |
|---|---|---|---|---|
| L-tier draft: 768 × 12 layers | 12.6M | ~105M | ~12% | ~92M |
| Same, UNtied | 25.2M | ~118M | ~21% | ~92M |
| GPT-2 small (50,257 vocab, 768×12, tied) | 38.6M | 124M | **31%** | ~85M |

Punchlines:
- Our 16k-vocab "105M" model has **more active params than GPT-2 small (124M)** — 92M vs 85M.
  Vocab choice quietly moved ~7% of capability-relevant budget. Small vocab + tying is why
  modern small models punch above their param count.
- At S-tier the effect is brutal: 16,384 × 384 = 6.3M embed vs ~10M total ≈ **63% embeddings
  if untied, ~35% tied**. Small models MUST tie; big models (70B+) barely care (embed <1%).

## Does "decent 100M model" mean increasing the count? Where is this decided?

**Decided in phase 3** — it was always the phase-3 param-budgeting deliverable, now made an
explicit decision point (spec updated; supersession of D-001 sizes tracked as RW-2). The framing:
- The question is not "100M vs 125M total" but "how many ACTIVE params and in what shape".
- Aspect ratio matters too: GPT-2-style wide-shallow (12×768) vs modern deep-narrow
  (e.g. 24×576) at the same budget — depth generally buys more at small scale (SmolLM,
  MobileLLM findings), at some MPS-throughput cost. Candidate P5-G ablation.
- With the cloud option (D-010), bumping L-tier to ~125–160M is affordable compute-wise;
  the real constraint is DATA (below).

## The coupling: params → tokens (why more model ⇒ more corpus)

Chinchilla (Hoffmann '22): compute-optimal ≈ **20 tokens per parameter**.
- 105M → ~2.1B tokens; 160M → ~3.2B.
- We have: 17.7M core tokens (books+dictionary) + ~500M raw TinyStories (not yet tokenized).
- Gap-closing options, in order: (a) repeat data — Muennighoff '23 shows up to ~4 epochs of
  repeated data is nearly as good as fresh (500M × 4 ≈ 2B effective ✓ for ~105M params);
  (b) add a FineWeb-Edu sample (needs a >2GB-download approval); (c) deliberately train
  under-budget and *measure* what undertraining looks like (legitimate lab result).
- Yes — this ripple was anticipated but is now formalized: **RW-1** (tokenize supplement, size
  depends on phase-3 choice) and **RW-2** (recompute time/cost if L grows) in PROGRESS.md.

## Corrected mental model

"N params" is a budget, not a spec. The spec is: **vocab size, tied?, d_model × layers
(aspect), FFN multiple** → which yields active params → which sets the token budget → which
sets corpus and wall-clock. Change any link and the chain re-propagates — that's what the
rework queue is for.

**Correction (phase 3, D-015):** this note's illustrative math used vocab=16,384 (reading "16k"
as 2^14). The tokenizer actually trained in phase 2 uses the literal 16,000 — see D-015's
correction note. The percentages/conclusions above are unchanged (16,384 vs 16,000 is a <2.5%
difference), but the exact embed-size figures (6.3M, etc.) are ~2% high; phase 3's configs and
D-015/D-016 use the correct 16,000.

## Related
D-001 (tiers, superseded-in-part by phase-3 outcome) · D-014 (16k vocab) · D-015 (tier sizes
finalized, vocab-number correction) · D-006/D-013 (corpus, supplement) · D-008/D-010 (compute
budget) · Papers: Kaplan '20, Chinchilla '22, Press & Wolf '16 (tying), Muennighoff '23
(data-constrained scaling), MobileLLM '24 (depth-vs-width at small scale).
