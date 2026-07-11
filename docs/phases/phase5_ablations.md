# Phase 5 — Ablation lab: research techniques, one variable at a time

**Goal:** implement and measure the techniques in `docs/TECHNIQUES.md`, wave by wave, producing
registered verdicts + figures. This phase IS the project's core research-learning payload.
**Effort:** open-ended; ~1 wave per session/chat. Order below is recommended but Waves are
independent (except C before F).

## Standing protocol (per EXPERIMENTS.md — re-read it)

- Tier S, ~50–100M tokens per run (pick once, keep constant), same seed & data order as
  `p4_s_baseline` unless measuring seed variance.
- **First task of the phase:** seed-noise study — baseline × 3 seeds → noise floor logged in
  EXPERIMENTS.md. Every later verdict must quote it.
- Each wave ends with: figure(s) saved to `docs/results/`, registry verdicts, notes.md files,
  and a 5-line summary appended to `docs/results/ablation_log.md`.
- Judge on val-loss @ equal tokens AND @ equal wall-clock, plus memory & tok/s. A run that's
  0.02 better loss but 30% slower is a *loss* at fixed compute — always say which axis wins.

## Wave A — Norms & activations (4 runs)
LayerNorm→RMSNorm; pre→post norm (expect instability — a *negative result on purpose*);
GELU→SwiGLU (match param count by shrinking ffn_mult to 8/3 — explain!); +QK-norm.
*Implement in `model/norms.py`/`ffn.py`; all are small diffs.*

## Wave B — Positional encodings (4–5 runs + probe)
learned (baseline) vs sinusoidal vs RoPE vs ALiBi vs NoPE. Extra: length-extrapolation probe —
train at 512, eval ppl at 1024/2048 per method (this is where RoPE/ALiBi shine; make the plot).

## Wave C — Attention variants (3–4 runs + inference bench) ⭐ DeepSeek flagship 1
MHA vs MQA vs GQA(2 groups) vs **MLA** (implement per DeepSeek-V2 paper §2: KV low-rank
compression to `kv_lora_rank`, decoupled RoPE keys; cache latent not K/V).
Measure: quality (val loss) AND **KV-cache bytes/token** (compute analytically + empirically)
AND generation tok/s at long context. MLA is the hardest implementation of the project —
budget a full session; write `notebooks/06_mla_explained.ipynb` with the matrix diagrams.

## Wave D — Optimizers & schedules (6–8 short runs) ⭐ flagship 2
AdamW baseline sweeps (wd 0/0.1; β2 0.95/0.999) → Lion → **Muon** (implement Newton–Schulz
orthogonalization; hidden-matrices-only, AdamW for embeddings/norms/head — follow the nanoGPT
speedrun recipe) → schedules: cosine vs **WSD** vs constant+warmup (WSD bonus: one stable run,
multiple decay branches = cheap multi-budget models — demonstrate!) → z-loss on/off →
batch-size study (0.06M/0.25M/1M tokens effective) → grad-clip off (watch it spike — screenshot).

## Wave E — Efficiency & memory (measurements more than quality runs)
bf16 vs fp32 (speed+memory+final loss); gradient checkpointing (mem↓ vs tok/s↓ curve);
micro-batch vs accumulation equivalence check; weight tying on/off; `torch.compile` attempt
(document whatever happens); activation-memory-vs-seq-len measured curve.

## Wave F — DeepSeek specials (3–5 runs) ⭐ flagship 3
- **DeepSeekMoE**: at S-tier: 8 fine-grained experts + 1 shared, top-2. Match *active* params
  to dense baseline. Aux-loss balancing vs **V3's aux-loss-free bias** method. Plot expert
  load histograms over training (the classic collapse-vs-balance picture).
- **Multi-Token Prediction (V3 §2.2)**: one extra sequential MTP head predicting t+2; loss =
  main + λ·mtp. Compare: val loss, and does the MTP head help or hurt at fixed compute?
- Note in writeup which V3 tricks we can't do on MPS (fp8, DualPipe) and what they're for.

## Wave G — Data & scaling (uses M tier too)
- Multi-epoch study on books-only: 1/4/16 epochs at fixed tokens — watch train/val gap open
  (overfitting lab).
- Dictionary ablation: with vs without dictionary in the mix → does "define X" eval improve?
  (needs P6 probes; can run later)
- **Domain-mix ablation (RW-4):** finance+wisdom share of the training stream at 10% vs 25%
  vs 50% (fixed total tokens, domain repetition ≤4 epochs) → domain probes vs general val
  loss. The specialization-vs-generality tradeoff, measured — the user's flagship data question.
- Mini scaling law: 5/10/25/50M params at fixed 200M tokens → fit L(N)=aN^-α + c, compare α
  to Chinchilla's. `notebooks/07_scaling_law.ipynb`.

## Learning checkpoints
Per wave, the user should be able to explain each technique's mechanism, its claimed benefit,
what we actually measured, and whether/why our small scale agrees with the paper.

## Exit criteria
≥ waves A–D done with verdicts (M2 milestone); figures in docs/results/; every run registered;
best-found recipe summarized in `docs/results/recipe.md` (feeds phase 9).
