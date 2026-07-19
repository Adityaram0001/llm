# Fine-tuning report (phase 8)

Before/after tables for the phase-8 fine-tuning experiments. Part A (SFT) is complete; Parts B
(LoRA) and C (DPO) will append their own sections.

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
