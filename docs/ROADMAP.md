# ROADMAP — LLM-Lab master plan

Goal: build a ~100M-param GPT from scratch on a MacBook Pro M4 (16GB), understand **every
decision** on the way, and use it as a lab to implement/compare training techniques from
research papers (DeepSeek line included). Learning > model quality. Every decision logged
(`DECISIONS.md`), every run registered (`EXPERIMENTS.md`).

Execution model: **one phase ≈ one or a few fresh Claude (Sonnet) chats.** Each phase has a
self-contained spec in `docs/phases/`. `CLAUDE.md` defines session protocol; `PROGRESS.md`
carries state between chats.

## The compute reality (read once, remember forever)

- 100M params in fp32: weights 0.4GB, +grads 0.4GB, +AdamW states 0.8GB ≈ 1.6GB — fits easily;
  **activations** and throughput are the real constraints on 16GB/MPS.
- Rough M4 expectation (phase 0 measures the truth): S-tier 10M ≈ 15–40k tok/s,
  L-tier 100M ≈ 2–6k tok/s at seq 512, bf16.
- Chinchilla-optimal for 100M ≈ 2B tokens ≈ weeks on the Mac (measured, D-008). Hence D-001
  (ablate at 10M ≈ 1–2h per run, confirm at 30–50M) **plus D-010**: big runs can burst to a
  rented RTX 5090 (~30–60× throughput, ~$1/hr — hero run overnight for ~$10–20). Playbook:
  `docs/CLOUD.md`. All code stays device-agnostic (mps/cuda/cpu) so local↔cloud is friction-free.

## Phases

| # | Phase | What you learn | Key deliverables |
|---|-------|----------------|------------------|
| 0 | Environment & MPS baseline | MPS vs CUDA, bf16/fp16/fp32, measuring FLOPs & tok/s | `.venv`, `verify_env.py`, `bench_mps.py`, measured budget |
| 1 | Corpus | data curation, cleaning, dedup, licensing, token counting | `data/clean/*.txt`, corpus stats notebook |
| 2 | Tokenizers | BPE algorithm internals, vocab-size trade-offs, fertility | scratch BPE + HF BPE + GPT-2 comparison, chosen tokenizer |
| 3 | Architecture | every tensor shape in a transformer; param budgeting | `llmlab.model`, S/M/L configs, shape-walkthrough notebook |
| 4 | Training engine | loss curves, LR schedules, grad accumulation, checkpointing, wandb | `train.py`, first real pretrains (S), compare notebook |
| 5 | Ablation lab | the research papers, one variable at a time | ~15 studies, figures, verdicts in registry |
| 6 | Evaluation | perplexity, downstream probes, generation quality, contamination | `llmlab.eval`, eval report for best models |
| 7 | Data factory | synthetic data generation, prompt engineering, validation | `tools/data_factory/` pipeline + first 2–5k Q&A pairs |
| 8 | Fine-tuning | SFT, loss masking, LoRA from scratch, DPO | chat-able fine-tuned model, before/after eval |
| 9 | Capstone | putting it together; writing up | 100M hero run with best-found recipe + final report |

Phases 5–7 can interleave; 7 can start any time after 2 since it only needs the tokenizer.

## Phase 5 studies (the research-paper menu — details in TECHNIQUES.md)

Grouped in study waves; each wave = one Sonnet chat, S-tier runs, one variable each:

- **Wave A — normalization & activations:** LayerNorm vs RMSNorm; post- vs pre-norm; GELU vs
  SwiGLU; QK-norm.
- **Wave B — positional encoding:** learned vs sinusoidal vs RoPE vs ALiBi vs NoPE (+ length
  extrapolation probe).
- **Wave C — attention variants:** MHA vs MQA vs GQA vs **MLA (DeepSeek-V2)**; KV-cache size
  & inference-speed measurements.
- **Wave D — optimizers & schedules:** AdamW (β, wd, eps sweeps) vs Lion vs **Muon**; cosine vs
  **WSD (warmup-stable-decay)** vs constant+warmup; batch-size & grad-clip effects; z-loss.
- **Wave E — efficiency:** grad accumulation trade-offs, gradient checkpointing, bf16 vs fp32,
  seq-len packing, weight tying, `torch.compile` (if it works on MPS).
- **Wave F — DeepSeek specials:** fine-grained **DeepSeekMoE** (shared + routed experts, aux-loss
  vs **aux-loss-free balancing** from V3), **Multi-Token Prediction** (V3).
- **Wave G — data & scaling:** multi-epoch overfitting study (books-only), dictionary-in vs
  dictionary-out (does the model define words better?), mini scaling law: 5M→10M→25M→50M at
  fixed tokens, fit the curve.

## Milestones (definition of "on track")

1. **M1 (end P4):** an S-tier model pretrained on books+dictionary generates recognizably
   English text; loss curves live in wandb; two runs compared in a notebook.
2. **M2 (mid P5):** ≥8 registered ablations with verdicts + noise floor known.
3. **M3 (end P6):** eval suite runs on any checkpoint in one command.
4. **M4 (end P8):** fine-tuned model answers "Define 'ephemeral'" style questions in chat format.
5. **M5 (end P9):** hero run done + `docs/results/final_report.md` comparing all techniques.
