# Fine-tuning report (phase 8)

Before/after tables for the phase-8 fine-tuning experiments. Parts A (SFT), B (LoRA), and C (DPO)
are all complete — see each section below.

## Part A — SFT (full fine-tune), 2026-07-19

**Run:** `20260719_p8_sft-s-dictionary` · **base:** `20260711_p4_s-baseline` (S-tier, 9.71M) ·
**data:** `data/sft/sft_dictionary_qa` (2916 train / 154 val grounded dictionary Q&A, phase-7
factory) · lr 2e-5, 3 epochs, bf16, assistant-only loss mask · ~1m44s on the M4.

### Before / after

| metric | base | SFT | delta |
|---|---:|---:|---|
| SFT val loss (assistant-masked CE) | 5.54 | **3.83** | learned the task |
| **stop-rate** (answers & stops ≤64 tok) | 0.0% | **80.0%** | +80 pts |
| mean answer length (tokens) | 64.0 (ran to budget) | 34.3 | concise, bounded |
| dict MC accuracy (chance = 25%) | 26.5% | 29.5% | +3.0 pts (≈retained) |
| dict cloze exact-match | 0.0% | 0.0% | — |
| **pretrain val ppl** (books+dict, forgetting) | 34.93 | 40.10 | **+14.8%** |

`definition_completion_ppl` is intentionally omitted from this table (RW-6: it is computed on
silently corrupted text until that eval bug is fixed). The other P6 probes (MC, cloze) use a
boundary shape RW-6 verified safe.

### What SFT actually did

- **Behavior, not knowledge.** The single biggest effect is the answer-vs-continue flip: the base
  model has no notion of "a turn ends" and simply continues in book-prose (stop-rate 0%); after SFT
  it answers in a definitional register and emits `<|endoftext|>` 80% of the time. Knowledge barely
  moved (MC 26.5%→29.5%, both near chance) — a 10M model can't be taught facts it never learned in
  pretraining; SFT teaches the *protocol*, not the content.
- **Catastrophic forgetting is real and visible.** Specializing on 2916 narrow examples cost +14.8%
  perplexity on the original books+dictionary distribution. At lr=2e-5 / 3 epochs this is mild; the
  low LR is the main guard (spec's 1e-5..5e-5 range). Mitigations to try in a later run: mix a slice
  of pretrain data into the SFT stream, fewer epochs, or LoRA (Part B — frozen base can't drift as far).

### Qualitative (greedy, chat-formatted prompt)

> **"What does ephemeral mean?"**
> *base →* "…what is not that it is not merely the best. For, as for example, if it is for the sake of what is necessary…" (ignores the question, continues a document)
> *SFT →* "An expression of substance, like the expression of a substance that is solved or used…" (answers, stops)

Content is nonsensical because the base is a 10M ablation model — the honest takeaway is that SFT
fixed *how* it responds, and there was little real knowledge underneath to surface. The mechanics
(loss masking, forgetting, chat template, REPL) are what phase 8 is here to teach; the phase-9
capstone (100M, real token budget) is where SFT lands on a model worth talking to.

### Artifacts
- code: `src/llmlab/data/{chat_format,sft_loader}.py`, `src/llmlab/train/{sft_config,sft_trainer}.py`,
  `scripts/{sft,eval_sft,chat}.py`, `configs/sft_s_dictionary.yaml`
- run: `experiments/20260719_p8_sft-s-dictionary/` (config, metrics.jsonl, samples/, eval_sft.json, notes.md)
- decision: D-051

## Part B — LoRA from scratch, 2026-07-19

**Runs:** all fine-tune `p4_s_baseline` on the same dictionary-QA data as Part A; the only variable
vs full FT is a frozen base + a rank-`r` adapter (`src/llmlab/train/lora.py`, `W + (α/r)·BA`,
B=0 at init so training starts bit-identical to the base). LoRA runs use lr **5e-4** (25× the
full-FT 2e-5 — few low-rank params, so a higher LR is both safe and needed). Rank sweep = r8 vs r32
(attn-only); placement study = r8 attn-only vs r8 attn+ffn. Each ~1.5 min on the M4.

| method | trainable | % | AdamW opt-state | ckpt (deliverable) | best val | stop-rate | pretrain ppl (adapted) |
|---|---:|---:|---:|---:|---:|---:|---|
| **full FT** | 9,713,472 | 100% | 116.6 MB | 116.7 MB† | 3.828 | 80% | 34.9→40.1 (+14.8%) |
| LoRA r8 attn | 184,320 | 1.9% | 2.2 MB | **0.77 MB** | 3.942 | 95% | 34.9→44.4 (+27.1%) |
| LoRA r32 attn | 737,280 | 7.6% | 8.9 MB | 2.98 MB | 3.924 | 95% | 34.9→43.4 (+24.2%) |
| LoRA r8 attn+ffn | 437,760 | 4.5% | 5.3 MB | 1.80 MB | **3.777** | 90% | 34.9→48.0 (+37.3%) |

†full-FT `best.pt` includes optimizer state; the shippable weights alone are ~39 MB. The LoRA
adapter *is* the whole deliverable.

### What the numbers say

- **Trainable params / optimizer memory — the real LoRA win.** AdamW keeps a gradient + two fp32
  moments per *trainable* param (`optimizer_state_bytes`). LoRA shrinks that from 116.6 MB to
  **2.2–8.9 MB (13–53×)**. At 10M params the absolute saving is small (this model trains fine
  either way); the same ratio at 7B+ is the difference between fitting one GPU or not — which is why
  LoRA is transformative there and merely tidy here. The forward FLOPs/activations are ~unchanged
  (LoRA adds only a tiny low-rank matmul), so **LoRA's win is memory, not speed** — LoRA tok/s was
  13–16K; attn+ffn is the slowest of the three because it adapts 105 layers vs 60 (more adapter
  matmuls). (Full-FT tok/s wasn't logged — the Part-A run predates the `tok_s` metric.)
- **Quality: LoRA is competitive-to-better here.** r8 attn+ffn (**3.777**) actually beat full FT
  (3.828); the rank bump r8→r32 helped a little (3.942→3.924), but **placement mattered more than
  rank** — adapting the FFN too (attn+ffn) beat quadrupling the rank (r32 attn). And every LoRA
  variant answered-and-stopped *more* reliably than full FT (90–95% vs 80%). Plausibly the
  low-rank constraint + higher LR fits this tiny SFT set well without letting the model wander.
- **Forgetting: honest nuance.** Measured on the *adapted* model, LoRA forgot **more** (+24–37% vs
  full FT's +15%) — but that is confounded by the 25× higher LR, and it is **fully reversible**: the
  base weights are frozen and bit-identical to the original `p4_s_baseline`, so removing the adapter
  restores pretrain ppl 34.9 exactly. Full FT can never recover its base. An LR-matched forgetting
  comparison is a flagged follow-up, not done this session.

### Artifacts
- code: `src/llmlab/train/{lora.py,sft_infer.py}`, `scripts/compare_finetune.py`,
  `configs/sft_s_dictionary_lora_{r8_attn,r32_attn,r8_attnffn}.yaml`, `tests/test_lora.py` (+10)
- runs: `experiments/20260719_p8_sft-s-dictionary-lora-*` (adapter-only checkpoints)
- data: `docs/results/finetune_partB.json`; decision: D-052

## Part C — DPO from scratch, 2026-07-19

**Run:** `20260719_p8_dpo-s-dictionary` · policy AND frozen reference both start from Part A's
full-FT SFT model (`20260719_p8_sft-s-dictionary`) · **data:** `data/dpo/dictionary_pairs`
(2740 train / 144 val preference triples) — chosen = the phase-7 SFT pairs reused as-is; rejected
= a new data-factory generation rotating three deliberate failure modes (**wrong_fact / verbose /
off_format**, DeepSeek v4-flash, $0.20 for 2884 pairs) · lr 5e-6, beta 0.1, batch_size 16,
max_len 640, bf16 · **stopped early at step 91/172 (53% of 1 epoch)** — see "why" below.

### Why the run was stopped early (not a crash — a deliberate call)

Per-step wall-clock degraded steeply and non-linearly over the run (~4.3s/step early -> ~15-30s/step
by step ~90), ruled out as a batch-shape artifact (checked directly: the rejected side's padded
width averages ~460 tokens with no upward trend across the visited steps) — most likely a
sustained-heavy-MPS-workload effect, since DPO runs **four** forward passes per step (policy x
{chosen, rejected}, reference x {chosen, rejected}) on long sequences, vs SFT's one short one.
Meanwhile the training signal had already saturated hard: by step 75's eval, val reward_accuracy
had risen 79% -> 95.8% and val reward_margin 0.04 -> 8.97 — a very large policy/reference
divergence for well under half an epoch. Rather than let an increasingly-slow run balloon toward
"multi-hour" without checking in first (CLAUDE.md's own rule), the run was interrupted with
SIGINT — `DPOTrainer` catches it cleanly (saves `latest.pt`, writes a real registry row); `best.pt`
(step 75, the actual checkpoint evaluated below) was unaffected.

### Before / after

| metric | base | SFT (pre-DPO) | DPO | delta (SFT->DPO) |
|---|---:|---:|---:|---|
| stop-rate (answers & stops ≤64 tok) | 0.0% | 82.0% | **99.0%** | +17 pts |
| mean answer length (tokens) | 64.0 | 33.9 | **16.5** | much shorter (see catch, below) |
| dict MC accuracy (chance = 25%) | 26.5% | 29.5% | 33.0% | +3.5 pts (small, noisy) |
| dict cloze exact-match | 0.0% | 0.0% | 0.0% | — |
| **pretrain val ppl** (books+dict, forgetting) | 34.93 | 40.10 (+14.8%) | **44.82 (+28.3% vs base)** | +11.8% further, in 91 steps |
| reward_accuracy vs frozen SFT ref (val) | — | 0.0%¹ | **95.8%** | the real DPO signal |
| reward_margin vs frozen SFT ref (val) | — | 0.0000 | **8.9694** | large, real preference shift |

¹ Degenerate: when policy == reference exactly, every log-ratio is 0 and the strict `>` in
`reward_accuracy` reports 0% by construction — NOT evidence SFT preferred the rejected answers.
`definition_completion_ppl` remains omitted (RW-6).

### The honest catch: length confound behind the short answers

A **reference-free** check — does a model, on its own, already assign a higher raw summed
log-prob to chosen over rejected? — says **yes, 95.8% of the time, even for the pre-DPO SFT
model** (mean gap +623.7 nats, rising to +713.4 after DPO). That is *not* evidence SFT already
understood the wrong_fact/verbose/off_format distinction: chosen responses average ~59 tokens,
rejected average ~461 (the verbose failure mode dominates that mean, and some pairs run up to
~10x longer), and an un-normalized summed log-prob is mechanically larger (less negative) for a
shorter sequence almost regardless of content — a **length confound**. DPO's actual reward is
computed relative to the reference on the SAME `y`, so it is length-invariant *by construction*;
but the training *gradient* has no such protection, and the qualitative samples
(`experiments/20260719_p8_dpo-s-dictionary/samples/step_000050.txt`) show the effect directly —
answers got terse and sometimes vague (`"What is a 'go-between'?" -> "A 'wit.'"`), consistent with
the model partly exploiting "shorter closes the reward gap" alongside genuinely learning to stop
rambling (the stop-rate win is real and separately explained). **This is a well-documented
real-world RLHF/DPO failure mode (verbosity/length bias in preference data)** — the honest
takeaway is that the failure-mode data generation should length-balance the rejected side (or the
loss should length-normalize) before trusting `reward_margin` as a pure quality signal on a
longer run.

### What DPO added on top of SFT, honestly

- **Real preference learning happened** — reward_accuracy 0%(degenerate)->95.8%, and the
  qualitative stop-rate win (82%->99%) is a genuine, length-unconfounded improvement: DPO
  specifically suppressed the off_format failure mode (rambling instead of answering).
- **But it came fast and partly via a shortcut** (length), and forgetting compounded quickly
  (+11.8% pretrain ppl in under half an epoch, on top of SFT's own +14.8%) — exactly the
  over-optimization risk the phase spec's "track reward margins & KL drift" instruction exists to
  catch. The notebook derivation's Section 8 predicted this pattern before the run confirmed it.
- **Not done this session** (flagged for a follow-up run): root-cause the MPS step-time slowdown;
  length-balance or length-normalize the preference data/loss; re-run to a complete epoch once
  both are addressed for a cleaner end-of-epoch number.

### Artifacts
- code: `src/llmlab/data/dpo_loader.py`, `src/llmlab/train/{dpo,dpo_config,dpo_trainer}.py`,
  `scripts/{dpo,eval_dpo,build_dpo_pairs}.py`, `configs/dpo_s_dictionary.yaml`,
  `tools/data_factory/{seeds.py (sft_pairs kind), tasks/dpo_dictionary_pairs.yaml}`,
  `tests/test_dpo.py` (+10, full suite 193 pass)
- run: `experiments/20260719_p8_dpo-s-dictionary/` (config, metrics.jsonl, samples/,
  eval_dpo.json, notes.md)
- derivation: `notebooks/09_dpo_from_scratch.ipynb` (RLHF objective -> closed-form optimal
  policy -> Bradley-Terry substitution -> the loss, executes cleanly, includes real-run results)
- decision: D-053
