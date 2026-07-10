# Phase 8 — Fine-tuning: SFT, LoRA, DPO

**Goal:** turn the best pretrained model into a small chat/Q&A model using data-factory data;
implement LoRA and DPO **from scratch** (no peft/trl — reading their source afterwards is the
epilogue); measure everything before/after.
**Effort:** 2–3 sessions. Requires phases 4, 6, 7.

## Part A — SFT (full fine-tune)

1. **Chat format**: use the reserved special tokens (`<|user|>`, `<|assistant|>`, `<|endoftext|>`).
   Write `llmlab/data/chat_format.py` (render + parse). Discuss why formats/templates matter
   (the user knows chat-ML style APIs — connect to that).
2. **`llmlab/data/sft_loader.py`**: tokenize `data/sft/*/train.jsonl`, pad or pack, build
   **loss masks so only assistant tokens contribute** (THE key mechanic of SFT — visualize a
   masked example in the notebook).
3. **`scripts/sft.py`** (reuses trainer with a different dataset+loss mask): S/M-tier from the
   best pretrain checkpoint. Low LR (1e-5..5e-5), 1–3 epochs, watch for catastrophic forgetting
   (track pretrain-val ppl alongside SFT loss — the divergence between them is the lesson).
4. Eval: dictionary probes (P6) before/after; + new **instruction-following battery** (does it
   answer vs continue the text?); side-by-side generations.
5. `scripts/chat.py` — minimal REPL to talk to the model. (The payoff moment. 🎉)

## Part B — LoRA from scratch

1. `llmlab/train/lora.py`: wrap chosen `nn.Linear`s with `W + (α/r)·BA`; freeze base; init
   A gaussian/B zero (explain why B=0); merge-back utility.
2. Repeat the Part-A SFT with LoRA (r=8/32 sweep, attn-only vs attn+ffn placement study).
3. Compare vs full FT: quality, trainable params, peak memory, tok/s, checkpoint size.
   At 100M params LoRA's *memory* win is modest (optimizer states) — compute and show the
   exact numbers; explain why it's transformative at 7B+ (connect to user's Gemma-12B experience).

## Part C — DPO

1. Data: preference pairs via data factory (chosen = good definition answer, rejected =
   subtly wrong/verbose/off-format — generate rejected variants deliberately).
2. `llmlab/train/dpo.py`: implement the DPO loss (policy vs frozen reference model,
   β·log-ratio margins) — derive it in the notebook first, ~30 lines of code after.
3. Train from the SFT checkpoint; track reward margins & KL drift; eval battery again;
   discuss what DPO changed vs SFT (style? correctness? verbosity?).

## Stretch — GRPO (only if user wants; from DeepSeek-Math/R1)
Verifiable-reward task: multiple-choice definition answering (reward = exact right option).
Group sampling, group-relative advantage, policy-gradient update. Even a partial attempt is
a great lesson in why RL training is finicky.

## Learning checkpoints
- Loss masking mechanics; catastrophic forgetting and its mitigations (LR, epochs, mixing
  pretrain data into SFT).
- LoRA math and where its savings actually come from; rank/α interplay.
- DPO objective from the RLHF problem statement; the role of the reference model & β.

## Exit criteria
Chat REPL demo works; before/after eval table in `docs/results/finetune_report.md`;
M4 milestone; all runs registered; decisions logged.
