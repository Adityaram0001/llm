# 20260719_p8_sft-s-dictionary-lora-r8-attnffn — LoRA SFT (r=8, attn+ffn), phase 8 Part B

LoRA fine-tune of `p4_s_baseline` on the same dictionary-QA data as the Part-A full FT
(`20260719_p8_sft-s-dictionary`). Only variable vs full FT: frozen base + rank-r adapter, lr 5e-4
(LoRA tolerates a higher LR — few low-rank params to move). 3 epochs, ~1.5 min on the M4.

- **Trainable params:** 437,760 (4.5%) of 9.71M — optimizer keeps grad+2 moments for only these.
- **Adapter checkpoint:** 1.80MB (vs full FT's 116.7MB best.pt) — the LoRA deliverable.
- **Best SFT val loss:** 3.777  (full FT: 3.828).
- **Instruction stop-rate:** 90%  (full FT: 80%).
- **Forgetting (adapted model):** pretrain ppl 34.9→ +37.3% — but **fully reversible**: the frozen
  base is bit-identical to the original (drop the adapter → ppl 34.9 back), unlike full FT.

Full cross-run comparison + analysis: `docs/results/finetune_report.md` (Part B), D-052.
Repro: `python scripts/sft.py --config configs/sft_s_dictionary_lora_r8_attnffn.yaml`
