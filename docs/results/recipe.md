# Recipe — phase 5's best-found configuration, for phase 9's L-tier hero run

Every choice below cites the decision (`D-xxx`) and/or run_id that established it. This is the
input `docs/phases/phase9_capstone.md` step 1 asks for — assemble `configs/model_l_hero.yaml` +
`configs/train_l_hero.yaml` from this, not from scratch.

## Architecture

| Axis | Choice | Why | Source |
|---|---|---|---|
| Norm | RMSNorm | Borderline vs LayerNorm (-0.0158, ~noise floor) but cheaper | D-036, Wave A |
| Norm position | Pre-norm | Post-norm stagnates by step 150 at this depth (not a blow-up) | D-036, Wave A |
| FFN | SwiGLU, `ffn_mult=8/3` | Clear, robust win over param-matched GELU (+0.17-0.2) | D-016, D-036 |
| QK-norm | **On** (new default) | Best result of Wave A, -0.062, gap widening over training — a genuine, non-obvious win worth carrying forward | D-036, Wave A |
| Positional encoding | RoPE (default) or **ALiBi** for long-context | RoPE beats learned/sinusoidal/NoPE at trained length; ALiBi beats RoPE AND *improves* under length extrapolation (ppl 32.56→31.67 @ 512→2048) — pick ALiBi if L-tier's chat-context goal needs extrapolation past its trained `max_seq_len` | D-037, Wave B; RW-5 (still open for L-tier's exact `max_seq_len`/pos_encoding pairing) |
| Attention | **MLA** (DeepSeek-V2) or GQA-2 if simplicity matters more | Quality flat across MHA/GQA/MQA/MLA (spread 2.6x noise floor); MLA gets ~3.2x smaller KV-cache at near-MHA quality (best cache/quality tradeoff), GQA-2 is the simpler ~2x-smaller-cache alternative if MLA's implementation complexity isn't worth it for the capstone | D-038, Wave C |
| MoE | Consider **DeepSeekMoE** (8 fine-grained routed + 1 shared, top-2) if total-param budget allows ~2x growth | Real win at matched ACTIVE params (-0.09 val_loss, >4x noise floor) — but ~2.18x slower tok/s (routing overhead), untested at fixed wall-clock (see Parking lot) | D-044, Wave F |
| MTP | Not yet justified | +0.017 vs control, at the noise floor's edge at S-tier/this token budget — revisit at L-tier/bigger budget before including | D-044, Wave F |
| Weight tying | Tied (keep D-016's default) | Untied showed a real win (-0.0278) but is NOT param-matched (+31.6% params) — doesn't cleanly overturn the tying-for-budget-efficiency argument | D-040, Wave E |

## Optimizer & schedule

| Axis | Choice | Why | Source |
|---|---|---|---|
| Optimizer | **Muon** (hidden matrices) + AdamW (embeddings/norms/head) | Single biggest effect found in the whole project (-0.1545 vs AdamW control, >10x noise floor), gap largest early but never closes | D-039, Wave D |
| Schedule | **WSD** (warmup-stable-decay) | Beats constant (-0.0674) which beats cosine (control) — WSD was already ahead before its own decay phase even started; bonus: multiple decay-budget forks off one shared stable checkpoint (real `--resume`, demonstrated) let you defer the final token-budget decision | D-039, Wave D |
| z-loss, wd, beta2 | Defaults are fine (wd=0.1, betas=[0.9,0.95], no z-loss) | All null results within noise at this budget | D-039, Wave D |
| Grad clip | Keep `grad_clip=1.0` | Off doesn't spike dramatically at this depth/warmup but is a real, steady degradation (+0.0215) | D-039, Wave D |
| Batch/accum factorization | Prefer the **largest micro-batch that fits** | Loss-invariant but NOT wall-clock-invariant — >2x spread between fastest/slowest factorization of the same effective batch | D-040, Wave E |

## Efficiency & memory

| Axis | Choice | Why | Source |
|---|---|---|---|
| Precision | **bf16 autocast** | Free ~35% speed win, zero quality cost | D-040, Wave E |
| `torch.compile` | **On, for CUDA runs** | Free ~18% speed win, zero quality cost, no graph-break issues at this scale — MPS remains unreliable (CLAUDE.md), don't depend on it there | D-040, Wave E |
| Gradient checkpointing | Only if memory-bound | ~1.72x peak-memory reduction at every seq_len, but costs ~27% speed — a real tradeoff, not a free win; use it if L-tier's longer `max_seq_len` needs the headroom | D-040, Wave E |

## Data & scaling (Wave G, D-045)

| Axis | Choice | Why | Source |
|---|---|---|---|
| Domain mix (RW-4, finance/wisdom flavor) | **10-25% of the training stream**, not 50% | Strictly monotonic general-val-loss cost as share rises (3.980→4.015→4.055→4.144 @ 0/10/25/50%); no sign of a plateau by 25%, so more than that trades away general quality faster than the flavor is worth | Wave G domain-mix ablation |
| Domain corpus size | Grow the 62-book/6.76M-token pool before scaling domain share up much at L-tier | L-tier's much bigger token budget would push domain repetition well past the spec's "≤4 epochs" guideline at even 25% share unless the raw pool grows too | Wave G, RW-4 |
| Repetition budget | Prefer **fresh tokens or ≤4x repetition** at L-tier, not the heavy repetition (~11.3 epochs) this wave used to keep S-tier runs fast | Wave G's own scaling-law finding: bigger models overfit a small repeated pool FASTER, not slower — repetition tolerance likely shrinks, not grows, as L-tier's ~100M params vastly exceeds every model tested here | Wave G scaling law |
| Mini scaling law | L(N) ≈ 11909.67·N^-0.694 + 3.102 (fit on best/early-stopped val_loss, 4 points 5-50M params, fixed 200M tokens, lr NOT retuned per size) | Steeper/noisier than Chinchilla's ~0.34 as expected from so few points/such a narrow range at fixed (not muP-scaled) lr — treat as a qualitative "returns diminish, and repetition tolerance shrinks with N" signal, not a precise extrapolation to L-tier's ~100M | Wave G scaling law |
| Multi-epoch tolerance | Watch the train/val gap, not just val_loss level | Val loss can plateau (not worsen) while the train/val gap still opens badly — the gap is the earlier, more sensitive overfitting signal | Wave G multi-epoch lab |

## What's still an open decision for phase 9

- **RoPE vs ALiBi + `max_seq_len` for L-tier** (RW-5, part b) — needs a deliberate choice once
  the capstone's real chat-context length target is fixed, not before.
- **MoE for the capstone** — real quality win, but untested at fixed WALL-CLOCK (only fixed
  token budget); the ~2.18x slower tok/s means it may not be worth it if training time (not
  token count) is the binding constraint. Resolve with an equal-wall-clock rerun before
  committing (see Parking lot in PROGRESS.md).
- **Exact domain-mix share for the capstone** (10% vs 25%, within the recommended range) — the
  user's call, a specialization-vs-generality tradeoff with no single "correct" answer.
- **Whether to grow the domain corpus** before the L-tier run, and by how much.
