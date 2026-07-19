# 20260719_p8_sft-s-dictionary — SFT (full fine-tune), phase 8 Part A

**What:** first supervised fine-tune of the project. Base = `20260711_p4_s-baseline`
(the ratified S-tier reference, 9.71M params, val_loss 3.5037). SFT data = phase-7 data factory
output `data/sft/sft_dictionary_qa/{train,val}.jsonl` (2916 train / 154 val grounded dictionary
Q&A pairs). Full fine-tune (all weights), lr 2e-5, 3 epochs (276 steps), bf16, assistant-only
loss mask, ~1m44s on the M4.

## Result — SFT changed *behavior*, not knowledge (as expected at 10M params)

| metric | base | sft | read |
|---|---|---|---|
| SFT val loss | 5.54 | **3.83** | learned the QA task/format |
| **stop-rate** (answers & stops within 64 tok) | **0%** | **80%** | THE headline: base never stops, SFT answers then emits `<\|endoftext\|>` |
| mean answer length (tok) | 64 (ran to budget) | 34.3 | concise answers, not a runaway document |
| dict MC accuracy (chance 25%) | 26.5% | 29.5% | knowledge ~retained, tiny bump — SFT can't teach facts a 10M base never had |
| dict cloze exact-match | 0% | 0% | reverse word-lookup beyond a 10M model either way |
| **pretrain val ppl** (books+dict) | **34.93** | **40.10** | **catastrophic forgetting: +14.8%** as the model specialized |

Before/after generations (`samples/`, `eval_sft.json`): the base model *continues in book-prose*
and ignores the question ("What does ephemeral mean?" → a Socratic ramble); the SFT model
*answers in a definitional register and stops* ("An expression of substance, like…"). Content is
nonsensical — a 10M model has no real definitional knowledge to surface — but the answer-vs-continue
behavior flip is dramatic and is the whole point of Part A.

## The two teaching artifacts this run makes concrete

1. **Assistant-only loss mask** (`llmlab/data/{chat_format,sft_loader}.py`): only assistant
   content + the stop token carry loss (`target = -1` elsewhere, honored by the model's existing
   `cross_entropy(ignore_index=-1)`). The mask is exact by construction — each turn's content is
   tokenized independently and marker IDs spliced in, so it never guesses a boundary from
   separately-encoded token counts (the fragile pattern behind eval bug RW-6).
2. **Catastrophic forgetting, live**: `SFTTrainer` measures frozen pretrain-val ppl at every eval
   alongside SFT loss. Watching SFT loss fall (5.54→3.83) while pretrain ppl climbs (34.9→40.1) is
   the divergence the phase spec asks students to *see*. lr=2e-5 kept it to +14.8%; a higher lr or
   more epochs would widen it (a Part-B/C follow-up: mix pretrain data into SFT to bound it).

## Repro
```
python scripts/sft.py --config configs/sft_s_dictionary.yaml
python scripts/eval_sft.py --sft-run experiments/20260719_p8_sft-s-dictionary
python scripts/chat.py --run experiments/20260719_p8_sft-s-dictionary   # talk to it
```
See D-051.
