# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phase 4 is **done** (milestone M1 declared). Phase 5
(`docs/phases/phase5_ablations.md`) is **in-progress**: seed-noise study done (D-035), **Wave A
done (D-036)**, **Wave B done (D-037)**, **Wave C done (D-038)**, **Wave D done (D-039)**, **Wave
E done (D-040)**. **Waves A-D done = the M2 milestone is DECLARED** (Wave E isn't part of M2's
exit criteria but is now done too). Wave F (DeepSeek specials: MoE aux-loss vs aux-loss-free,
MTP) is next — the flagship-3 wave, needs new model code (routing, MTP head).

**This session (2026-07-13, Wave E):** implemented the efficiency/memory knobs Wave E needed
(none existed before): `precision` (bf16/fp32) and `gradient_checkpointing` on `TrainConfig`/
`Trainer`/`GPT` (checkpointing wraps each block in `torch.utils.checkpoint.checkpoint`, gated on
training + no KV cache), `compile` (`torch.compile(model)`, guarded try/except; checkpointing
routed through a new `Trainer._raw_model` reference so save/load never depends on the compiled
wrapper's state_dict key-naming). Ran 6 short S-tier runs on the RTX 5090 (~28 min wall-clock)
plus a standalone `scripts/bench_activation_memory.py` seq_len sweep (new script). **Result
(D-040): four of five axes are NULL results on loss by design (efficiency knobs shouldn't change
what's computed) — the real findings are speed/memory numbers.** bf16 and torch.compile are both
free speed wins (~35% and ~18% respectively vs their disabled state, zero quality cost).
Gradient checkpointing costs ~27% speed at this size but gives a consistent **~1.72x peak-memory
reduction at every seq_len**, buying one more doubling of context before OOM on the 5090's 32GB.
Micro-batch/grad-accum factorization is loss-invariant (as it should be) but NOT wall-clock-
invariant — over 2x spread between the fastest (mb=128/accum=1) and slowest (mb=32/accum=4)
factorization of the identical effective batch, confirming D-022's launch-overhead-bound finding
and giving a concrete rule (prefer the largest micro-batch that fits). Weight tying off shows a
real quality win (-0.0278) but is honestly flagged as NOT param-matched (+31.6% params) so
doesn't settle the tying-vs-quality question cleanly — a param-matched rerun is a flagged
follow-up, not done this wave. New: `precision`/`gradient_checkpointing`/`compile` fields
(`train/config.py`), `_autocast`/`_raw_model`/`compile_status` (`train/trainer.py`),
`gradient_checkpointing` attribute + block-wrap (`model/gpt.py`), `scripts/
bench_activation_memory.py`, `scripts/plot_wave_e.py`, 6 `configs/train_s_wave_e_*.yaml` +
`configs/model_s_notie.yaml`, `docs/results/wave_e_efficiency_memory.png`, `docs/results/
wave_e_activation_memory{,_gradckpt}.csv`, +6 tests (89 local / 64 remote-cuda, all pass).
Also fixed (not a decision, logged in D-040's note): a trailing-slash rsync bug that briefly
created a stray incomplete `llmlab/` package on the remote pod, shadowing the real one and
breaking test collection — cleaned up, no project code affected.
**GPU LEFT RUNNING (singapore-b:25864) — user must stop it to stop billing** once satisfied
everything synced down correctly; nothing else pending on it this session.

**Earlier same day (Wave D):** implemented **Muon** (Jordan '24 Newton-Schulz
orthogonalization, hybrid with AdamW for embeddings/norms per the nanoGPT speedrun recipe) and
**Lion** (Chen '23 sign-based update) as new optimizers (`src/llmlab/train/optimizers.py`),
generalized `Trainer` to a list-of-optimizers design, generalized the lr schedule to dispatch
`cosine`/`wsd`/`constant`, and added PaLM z-loss. Ran 13 short S-tier runs on the RTX 5090
(~42 min wall-clock for the first 11, then 2 more for the WSD-fork bonus). **Result (D-039):
Muon is the single biggest effect found in the project so far** (-0.1545 val_loss vs the AdamW
control, >10x the D-035 noise floor) — gap largest early, narrowing but never closing (matches
Muon's "faster convergence" framing). **Schedule hierarchy WSD > constant > cosine** (-0.1213 /
-0.0674 / control) — *when* you decay matters as much as whether you decay at all; WSD was
already ahead of cosine before its own decay phase even started. **WSD multi-budget bonus**: two
decay forks off the SAME shared stable-phase checkpoint (`wave_d_constant`'s final weights, real
`--resume`, not simulated) at +10%/+26.7% tokens reached 3.3220/3.2768 — demonstrating you can
decide the final token budget after training, not before. Honest confounds flagged rather than
hidden: Lion's +0.4226 "loss" reflects one un-tuned paper-recipe hyperparameter guess, not a real
verdict against Lion; the batch-size study's 1M-tok/step point is partly confounded by an
unscaled 30-step warmup eating 32% of its 94-step budget (the cleaner 0.25M point confirms the
same direction). grad-clip-off did NOT spike as the spec predicted — `clip_grad_norm_` always
logs the pre-clip norm so the metric can't show a difference, and the real effect is a small,
steady degradation, not a blowup, at this depth/warmup. New: `optimizers.py` (`Lion`, `Muon`,
`zeropower_via_newtonschulz5`), hybrid-optimizer checkpointing, `_schedule_multiplier`, z-loss in
`train_step`, 13 `configs/train_s_wave_d_*.yaml`, `scripts/plot_wave_d.py`,
`docs/results/wave_d_optimizers_schedules.png`, +15 tests (96 local / 66 remote-cuda, all pass).
**GPU LEFT RUNNING (singapore-b:25864) — user must stop it to stop billing** once satisfied
everything synced down correctly; nothing else pending on it this session.

**Earlier same day (Wave C):** implemented **MLA** (`MLAAttention`, DeepSeek-V2 §2) + a
full **incremental KV-cache decode path** for all 4 attention variants (new
`src/llmlab/model/kv_cache.py`; `cache=` threaded through attention/block/gpt; `generate()`
rewritten prefill-once-then-1-tok/step — cached decode bit-exact vs full forward on cpu/mps/cuda).
Ran the 4-run Wave C ablation on the RTX 5090 (~50 min, singapore-b:25864). **Result (D-038):
quality is flat across MHA/GQA/MQA/MLA (spread 0.039 ≈ 2.6× the noise floor) — so cache decides.**
GQA-2 (−0.0205, 2× smaller cache) and MLA (−0.0166, 3.2× smaller) both marginally beat the MHA
control; MQA is the only real quality loss (+0.0186) but smallest cache. **MLA reproduces
DeepSeek-V2's headline at 10M params** (near-MQA cache, near-MHA quality). Honest tok/s caveat:
at this scale decode is launch-overhead-bound so the cache doesn't speed latency and MLA is ~25%
slower/tok (no absorption trick) — cache win is memory not latency. New: `MLAAttention`,
`kv_cache.py`, `scripts/bench_inference.py`, `scripts/plot_wave_c.py`,
`notebooks/06_mla_explained.ipynb` (matrix diagrams, executes clean), 4 model + 4 train configs,
`configs/model_s_attn_*` / `train_s_wave_c_*`, +10 tests (82 pass). Figures:
`docs/results/wave_c_attention_variants.png`, `docs/results/wave_c_inference_bench.csv`.
**GPU LEFT RUNNING — user must stop the singapore-b:25864 instance to stop billing** (checkpoints
for the 4 runs are still on it if ever needed; nothing else pending there). Important design note
carried into DECISIONS: the wave runs at **n_heads=4** (GQA-2 undefined at the baseline's 3
heads), so `20260713_p5_s-wave-c-mha` is the wave's control, not `p4_s_baseline`.

**Earlier — 2026-07-11 evening through 2026-07-12** — built the whole training engine
(deliverables 0b, 1, 2, 3, 3b), ran the first real experiments including an unattended overnight
lr-sweep + baseline pipeline, then reviewed the results.
**Same-day update (2026-07-12, phase 5 start):** ran the phase-5 seed-noise study (2 more seeds
on top of the existing baseline), then Wave A (4 runs) and Wave B (4 runs + length-extrapolation
probe), all on the RTX 5090 gpuhub instance (left up since the D-034 benchmark session — **user
confirmed shutting it down at session end, it is NOT running as of this update**, re-provision
from the saved "genesis" image next time, per D-029/CLOUD_GPUHUB.md). Noise floor: mean val_loss
3.5043, std 0.0062, **spread 0.0150** across seeds 1337/1338/1339 (D-035) — this is also the
first real (non-sweep) confirmation that a full training loop runs correctly on gpuhub's CUDA
hardware (~126K tok/s, ~13min/run vs Mac's 2.4hr). **Wave A (D-036):** RMSNorm≈LayerNorm
(borderline), pre-norm≫post-norm (post-norm stagnates, doesn't blow up), SwiGLU beats GELU
(param-matched, real), **+QK-norm is a real, robust win** (best of the wave, recommend as new
default). **Wave B (D-037):** required a real code fix first — RW-5's `GPT.forward()` guard now
only blocks learned/sinusoidal past `max_seq_len`, not rope/alibi/none (new
`scripts/eval_extrapolation.py` for any future length-probe work). Results: learned/sinusoidal/
NoPE all real-worse than RoPE at trained length; **ALiBi real-better than RoPE AND the
length-extrapolation probe is the project's cleanest paper reproduction yet** — ALiBi's val_loss
*improves* with more context (ppl 32.56→31.67 @ 512→2048) while RoPE degrades gracefully
(33.24→45.68) and NoPE collapses (40→732). Both waves' model-code axes needed **zero new
implementation** beyond RW-5's one-line guard fix — everything else was already wired in phase 3;
this is purely config+run+analysis work, which is why 2 full waves fit in one session.

Built: `src/llmlab/data/loader.py` (`MixedSourceLoader`/`Source` — memmap random-offset
sampling, stateless given `(seed, step)` so resume needs no sampler state, per-source mixing
weights + optional doc-boundary-respecting mode for RW-4 later); `src/llmlab/train/config.py`
(`TrainConfig` + nested dataclasses) and `src/llmlab/train/trainer.py` (`Trainer`: param groups,
warmup+cosine lr schedule, grad accumulation/clipping, eval loop, text sampling, checkpointing,
metrics.jsonl+wandb logging, graceful Ctrl-C, registry row); `scripts/train.py` (CLI, run-folder
creation, `--resume`, `--device` override); `scripts/find_batch_size.py` (D-018 calibration).
Configs: `configs/train_s_{baseline,smoke,cpu_canary,lr_sweep_{lo,mid,hi}}.yaml`. Tests:
`tests/test_loader.py` (7 tests), `tests/test_trainer.py` (3 tests) — full suite 61 passed.

**Decisions logged:** D-021 (baseline hyperparameters: lr 1e-3, effective batch ~64K tokens,
eval every 100 steps), D-022 (real MPS throughput for the S-tier model is flat ~11K tok/s across
micro_batch 1-32, not D-008's ~20.8K dummy-model estimate — kept micro_batch=16 anyway since
larger is free when flat; also fixed a list-aliasing bug in `find_batch_size.py`'s plateau
detection), D-023 (two real trainer bugs found via an actual kill+resume test, not just unit
tests: `wandb.init()` was silently swallowing SIGINT, and a step-checkpointing off-by-one made
resume replay — and double-apply the gradient update for — the last completed step; both fixed
and reverified bit-exact), D-024 (overnight lr-sweep-then-baseline automation), **D-025
(the sweep's result reviewed: D-021's lr=1e-3 ratified, not overridden — see below)**.

**All experiments run/registered/reviewed this session:**
- `20260711_p4_cpu-canary` — deliverable 0b portability canary (`--device cpu`), passed.
- `20260711_p4_s-smoke` — 150 steps, loss 9.69→5.38, samples already show dictionary-entry
  formatting.
- `20260711_p4_resume-test` — real `kill -INT` + `--resume`, bit-exact reproduction verified
  after the D-023 fixes (full bug story in its notes.md).
- `20260711_p4_s-lr-sweep-{lo,mid,hi}` (lr 3e-4/1e-3/3e-3, 300 steps each) — **mid (1e-3) won,
  strictly ahead of both alternatives at every logged checkpoint**, not just at the end; lo was
  undertrained (not unstable, just slower); hi didn't diverge (`grad_clip=1.0` held) but was
  consistently worse despite ending with a *lower* mean grad_norm than mid — a real lesson that
  clipping bounds the damage from a bad lr, not the outcome. See D-025 and each run's notes.md.
- `20260711_p4_s-baseline` — **THE S-tier reference run**, lr=1e-3 (ratified default), 1500
  steps / 98.3M tokens, val_loss 9.55→**3.5037** (perplexity 33.2), textbook power-law loss
  curve. Samples pick up the corpus's Socratic-dialogue register specifically by step 800 (see
  notes.md for the actual generated text) — legible evidence the model is learning from *this*
  corpus, not generic English. One open observation for phase 6: the dictionary-format prompt's
  output drifts toward book-prose by later checkpoints, plausibly because dictionary entries are
  a small minority of the S-tier corpus — worth a phase-6 eval probe.

All four registry rows now have real verdicts (not the auto-generated "review and fill in
notes.md" placeholder) and real notes.md conclusions.

`notebooks/05_compare_runs.ipynb` executes cleanly; re-run it now that the lr-sweep/baseline
runs exist (last executed mid-pipeline, so sections 4 still show the "skipping" message from
before the sweep/baseline landed — cosmetic only, the data is all there in metrics.jsonl).

**Exit criteria check (`docs/phases/phase4_training.md`):** baseline finished & registered ✅;
samples read as English-ish ✅ (Socratic-dialogue prose by step 800); resume verified ✅ (D-023,
bit-exact); comparison notebook renders ✅ (re-run for fresh plots, not required for the
criterion itself). **Milestone M1 can be declared.**

**Update 2026-07-12 (later same day):** RW-1 is now fully done — R2 bucket `llm` created by the
user, rclone installed + `.env` wired (D-026), tokenized data pushed and verified (2.879 GiB,
16 files). RW-3's other sub-steps (GitHub remote, Docker Hub, pod template) remain open, still
not needed for any S-tier work. RW-4 (domain corpus expansion) still needs the user to pick
titles; the loader's per-source mixing-weight design (`MixedSourceLoader`) was built
general-purpose with RW-4 in mind, so it shouldn't need a rewrite when that happens.

## Phase status

| Phase | Name | Spec | Status |
|-------|------|------|--------|
| 0 | Environment & MPS baseline | `docs/phases/phase0_setup.md` | done |
| 1 | Corpus: books + dictionary | `docs/phases/phase1_data.md` | done |
| 2 | Tokenizers (scratch + HF) | `docs/phases/phase2_tokenizer.md` | done |
| 3 | Model architecture | `docs/phases/phase3_architecture.md` | done |
| 4 | Training engine + first pretrain | `docs/phases/phase4_training.md` | done |
| 5 | Ablation lab (research techniques) | `docs/phases/phase5_ablations.md` | in-progress |
| 6 | Evaluation suite | `docs/phases/phase6_evaluation.md` | todo |
| 7 | Data factory (DeepSeek-assisted) | `docs/phases/phase7_data_factory.md` | todo |
| 8 | Fine-tuning: SFT / LoRA / DPO | `docs/phases/phase8_finetuning.md` | todo |
| 9 | Capstone: 100M hero run + report | `docs/phases/phase9_capstone.md` | todo |

## Phase 0 checklist (done)

- [x] `scripts/setup.sh` run: `.venv` created, requirements installed, `llmlab` editable install
- [x] `scripts/verify_env.py`: MPS available, bf16 autocast works, seed utility works
- [x] `scripts/bench_mps.py`: measured matmul TFLOPS + tokens/sec on a dummy ~9.1M-param transformer
- [x] Throughput numbers recorded in `docs/DECISIONS.md` (D-008; sets the compute budget for everything)
- [x] `notebooks/00_mps_playground.ipynb`: tensors on mps, autocast dtypes, sync timing pitfall, memory readout — executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated; phase marked done

## Phase 1 checklist (done)

- [x] `configs/corpus.yaml`: 112 books (20 user-picked authors + ~90 auto-selected from
  Gutenberg's catalog metadata, see D-011), GCIDE dictionary config, TinyStories supplement flag
- [x] `src/llmlab/data/acquire.py` + `scripts/build_corpus.py`: idempotent download → clean →
  dedup → stats pipeline (downloads cached in `data/raw/`, safe to re-run)
- [x] Gutenberg boilerplate stripped, unicode NFC-normalized, whitespace collapsed,
  exact-duplicate paragraphs deduped (hash-based) — all books clean in `data/clean/books/`
- [x] GCIDE dictionary parsed (119,984 entries) into `data/clean/dictionary_prose.txt`
  (bold-term template) + `data/clean/dictionary.jsonl` (structured, for phase 6/7 eval probes)
- [x] TinyStories supplement streamed to `data/clean/supplement/tinystories.txt` (D-013)
- [x] Held-out val split by whole document: `data/clean/val/books/{boethius,epictetus}...txt` +
  2% of dictionary entries in `data/clean/val/dictionary.jsonl` — never seen in training
- [x] `data/clean/manifest.json`: source URL, license, sha256, char/word counts per file
- [x] `notebooks/01_corpus_stats.ipynb`: composition, chars/4 vs GPT-2-calibrated token
  estimates, length histogram, common-words sanity check — executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated (D-011, D-012, D-013); phase marked done

## Phase 2 checklist (done)

- [x] `src/llmlab/tokenizer/bpe_scratch.py`: pure-Python byte-level BPE (train/encode/decode),
  supports `pretok_mode` in {none, whitespace, gpt2}
- [x] `notebooks/02_bpe_from_scratch.ipynb`: trained on `marcus-aurelius-meditations.txt`,
  shows first merges, pretokenization comparison, vocab-size-vs-compression curve, byte-level
  no-OOV demo (emoji/CJK/tags round-trip) — executes cleanly end to end
- [x] `src/llmlab/tokenizer/train_hf.py`: HF `ByteLevelBPETokenizer` trained on the full
  S-tier corpus at 8k/16k/32k, saved to `data/tokenized/tokenizers/hf_bpe_{8k,16k,32k}/`
  (`tokenizer.json` + `vocab.json`/`merges.txt`); special tokens `<|endoftext|>`, `<|pad|>`,
  `<|user|>`, `<|assistant|>` reserved for phase 8
- [x] `notebooks/03_tokenizer_compare.ipynb`: fertility/compression, vocab utilization,
  rare-word splitting, numbers/punctuation, embedding-table cost math, for scratch-bpe-8k /
  hf-bpe-8k/16k/32k / gpt2-50k — figures + written verdict — executes cleanly end to end
- [x] Decision logged: **D-014, HF BPE 16k vocab** chosen (user reviewed the comparison
  table); 5 comparison rows registered in `experiments/registry.csv` (p2, non-training rows)
- [x] `scripts/tokenize_corpus.py`: encodes train+val corpus → `data/tokenized/hf_bpe_16k/
  {train,val}.bin` (uint16 memmap) + `meta.json` (vocab size, per-doc token offsets, token
  counts); verified via decoding random slices. 17,665,275 train tokens (111 docs), 179,655
  val tokens (3 docs)
- [x] PROGRESS.md + DECISIONS.md updated (D-014); phase marked done

## Phase 3 checklist (done)

- [x] `src/llmlab/model/config.py`: `ModelConfig` dataclass (+ `MLAConfig`/`MoEConfig`/
  `MTPConfig`), `from_yaml`, validates `n_heads % n_kv_heads == 0` and MLA needs an `mla:` block
- [x] `norms.py` (LayerNorm/RMSNorm), `positional.py` (learned/sinusoidal/RoPE/ALiBi + relative-
  shift math), `attention.py` (MHA/GQA/MQA via SDPA, qk_norm, RoPE injection), `ffn.py`
  (GELU/SwiGLU), `block.py` (pre/post-norm residual wiring), `gpt.py` (embeddings→blocks→
  final norm→head; `forward`, `generate` w/ temperature+top-k+top-p, `num_params(breakdown=)`,
  `estimate_flops_per_token`)
- [x] `attention="mla"`, `moe`, `mtp` raise `NotImplementedError` (config fields exist, phase 5)
- [x] Tier sizes finalized vocab-aware, deep-narrow L-tier, FineWeb-Edu data-budget plan (D-015);
  baseline defaults tying/head_dim/dropout/init (D-016); `configs/model_{s,m,l}.yaml` committed
- [x] `tests/test_model.py`: 51 tests green on mps AND cpu
- [x] `notebooks/04_shapes_walkthrough.ipynb`: executes cleanly end to end
- [x] PROGRESS.md + DECISIONS.md updated (D-015, D-016); phase marked done

## Phase 4 checklist (done)

- [x] 0a. Data prep (RW-1): TinyStories + FineWeb-Edu tokenized to
  `data/tokenized/hf_bpe_16k/supplement_{tinystories,fineweb}.bin` (+ docstarts `.npy`); D-019
  bug fix (ambiguous story boundaries) + D-020 (FineWeb-Edu sizing) logged. R2 push (bucket
  step) deferred — blocked on RW-3, not required for S-tier work.
- [x] 0b. Portability smoke test (`--device cpu` canary) — `20260711_p4_cpu-canary`, passed
- [x] 1. `src/llmlab/data/loader.py` (memmap sampler + per-source mixing weights) — `MixedSourceLoader`/`Source`, 7 tests
- [x] 2. `src/llmlab/train/trainer.py` — built + two real bugs found/fixed via live resume test (D-023)
- [x] 3. `scripts/train.py`
- [x] 3b. `scripts/find_batch_size.py` (D-018) — real S-tier MPS numbers in D-022 (list-aliasing bug fixed)
- [x] 4. First experiments, all registered with real verdicts: `p4_smoke` (loss 9.69→5.38),
  resume test (D-023, bit-exact verified), `p4_s_lr_sweep_{lo,mid,hi}` (1e-3 won at every
  checkpoint, D-025), `p4_s_baseline` (1500 steps, val_loss 3.5037/ppl 33.2, D-025)
- [x] 5. `notebooks/05_compare_runs.ipynb` — executes cleanly, includes a numbers-grounded
  "reading a loss curve" section; sections 4 (lr sweep) and the baseline cell will populate once
  the overnight pipeline's runs exist

## Phase 5 checklist (in-progress)

- [x] Seed-noise study: `20260712_p5_s-seed-{1338,1339}` + reused `20260711_p4_s-baseline` as
  seed 1/3 → noise floor mean 3.5043, std 0.0062, **spread 0.0150** (D-035, logged in
  `docs/EXPERIMENTS.md`). Ran on the RTX 5090 gpuhub instance (`scripts/cloud/remote.env`, port
  25864 — **shut down at this session's end (user confirmed), NOT running/billing anymore**;
  `remote.env` will need updating with a new host/port once a fresh instance is provisioned —
  re-provision from the saved "genesis" image, D-029/CLOUD_GPUHUB.md).
- [x] Wave A — Norms & activations (4 runs, D-036): RMSNorm→LayerNorm **borderline** (-0.0158,
  at the noise floor, RMSNorm kept for compute cost); pre→post norm **negative result as
  predicted** (stagnates ~loss 6.8 by step 150, degenerate samples, NOT a blow-up — grad_norm
  stayed ≤1.52); SwiGLU→GELU (param-matched) **real, robust loss** (SwiGLU wins by ~0.17-0.2,
  confirms D-016); **+QK-norm real, robust WIN** (-0.062, gap widening over training — best of
  the wave, a genuine surprise, recommend as new default going forward). Figure:
  `docs/results/wave_a_norms_activations.png`. Summary: `docs/results/ablation_log.md`.
- [x] Wave B — Positional encodings (D-037): learned **real, worse** (+0.227, cannot
  extrapolate past 512 by construction); sinusoidal **real, WORST of the wave** (+1.486, a
  surprise — notably worse than even learned); **ALiBi real, BEST of the wave** (-0.021 at
  trained length, AND val_loss **improves** with more context: ppl 32.56→32.08→31.67 at
  512→1024→2048 — clean small-scale reproduction of the paper's headline claim, RoPE degrades
  33.24→36.79→45.68 by comparison); NoPE **real, worse + catastrophic under extrapolation**
  (ppl 40→67→732). Required a real code fix first (RW-5, partially resolved): `GPT.forward()`'s
  `max_seq_len` guard now only applies to learned/sinusoidal, not rope/alibi/none — see
  `src/llmlab/model/gpt.py`, `tests/test_model.py`, new `scripts/eval_extrapolation.py`. Figure:
  `docs/results/wave_b_positional_encodings.png`.
- [x] Wave C — Attention variants (MHA/MQA/GQA/MLA) + KV-cache-bytes + gen tok/s (D-038): quality
  flat across all 4 (spread 0.039 ≈ 2.6× noise floor) → cache decides; GQA-2 −0.0205 @2× smaller
  cache, MLA −0.0166 @3.2× smaller (reproduces DeepSeek-V2), MQA +0.0186 @4× smaller (only real
  quality loss). MLA + incremental KV-cache decode implemented & tested (bit-exact cpu/mps/cuda);
  `notebooks/06_mla_explained.ipynb` + `scripts/bench_inference.py` done. Honest caveat: at 10M
  params decode is launch-bound so cache ≠ speed, MLA ~25% slower/tok (no absorption trick).
- [x] Wave D — Optimizers & schedules (D-039): **Muon best of the wave** (-0.1545 vs AdamW
  control, >10x noise floor, gap largest early/narrowing but never closing) — new
  `src/llmlab/train/optimizers.py` (`Muon` Newton-Schulz hybrid w/ AdamW for embed/norms, `Lion`).
  **Schedule hierarchy WSD (-0.1213) > constant (-0.0674) > cosine (control)** — decaying only at
  the end beats never decaying, which beats cosine's continuous early decay. **WSD multi-budget
  bonus**: 2 real `--resume` decay forks off `wave_d_constant`'s shared checkpoint (+10%/+26.7%
  tokens) reached 3.3220/3.2768. z-loss + AdamW wd/beta2 sweep: null results (within noise) at
  this budget. grad-clip-off: real but undramatic (+0.0215, no spike — `clip_grad_norm_` always
  logs the pre-clip norm). batch-size study: fixed-token-budget bigger batch undertrains with lr
  not rescaled (confirmed, though the 1M-tok/step point has an unscaled-warmup confound). Lion's
  result flagged as untuned, not a real verdict against it. Figure:
  `docs/results/wave_d_optimizers_schedules.png`. 13 runs registered, +15 tests (96 pass).
- [x] Wave E — Efficiency & memory (D-040): 6 S-tier runs + a standalone memory-sweep benchmark,
  new code (`precision`/`gradient_checkpointing`/`compile` on `TrainConfig`/`Trainer`/`GPT`).
  4/5 axes NULL on loss by design (efficiency knobs, shouldn't change what's computed) — real
  findings are speed/memory: **bf16 and torch.compile are free speed wins** (~35%/~18% faster
  than disabled, zero quality cost); **gradient checkpointing** costs ~27% speed at this size for
  a consistent **~1.72x peak-memory reduction** at every seq_len (buys one more context-length
  doubling before OOM on the 5090); **micro-batch/accum factorization is loss-invariant but NOT
  wall-clock-invariant** (>2x spread between fastest/slowest factorization of the same effective
  batch — always prefer the largest micro-batch that fits); **weight tying off** shows a real
  quality win (-0.0278) but isn't param-matched (+31.6% params), so doesn't settle the question
  cleanly (flagged follow-up). Figure: `docs/results/wave_e_efficiency_memory.png`.
- [ ] Wave F — DeepSeek specials (MoE w/ aux-loss vs aux-loss-free, MTP) — needs RW-5 fixed for
  full context-length work but MoE/MTP themselves don't require it
- [ ] Wave G — Data & scaling (multi-epoch overfitting lab, dictionary ablation, RW-4 domain-mix
  ablation, mini scaling law `notebooks/07_scaling_law.ipynb`)
- [x] Exit criteria (M2): waves A-D done + verdicts, figures in `docs/results/`, all runs
  registered. **M2 DECLARED 2026-07-13.** `docs/results/recipe.md` (phase 9 input) still not
  written — deferred until Waves E-G land too, same reasoning as Wave C's session note.

## Rework queue (see CLAUDE.md "Change management")

| ID | What | Why | Fix in phase | Status |
|----|------|-----|--------------|--------|
| RW-1 | Tokenize TinyStories supplement + a FineWeb-Edu sample with hf_bpe_16k → `data/tokenized/hf_bpe_16k/supplement_*.bin`. **Fully done 2026-07-12**: tokenized (D-019, D-020: 520.5M + 992.8M tokens) AND pushed to R2 (D-026) — `r2:llm/data/tokenized/` now has all 16 files (train/val, both supplements + docstarts, meta.json, all 3 tokenizer vocabs), 2.879 GiB, verified via `rclone lsf -R` | D-015: L-tier is 105M, needs ~2.1B tokens; repetition alone (~4 epochs of core+TinyStories) was right at the edge, so a FineWeb-Edu sample was added for margin + topic diversity | 4 | done |
| RW-3 | One-time cloud accounts setup. **Done:** GitHub remote, R2 bucket + rclone (D-026), Docker Desktop installed locally, $10 gpuhub credit purchased, provider decision (D-027: gpuhub, native image-snapshot workflow, RunPod kept documented-but-unbuilt). **Cloud pipeline validated live end-to-end 2026-07-12 (D-029)**: RTX 4080 Super dry-run instance (D-028), `scripts/cloud/gpuhub_setup.sh` ran clean over SSH, real training smoke test passed (99,554 tok/s), checkpoint round-tripped CUDA→Mac-MPS. Image saved as **"genesis"** — contains OS+deps+conda env+our SSH key+`.env` (system disk only; repo/data live on the data disk, NOT in the image — user confirmed the `.env`-in-image finding is an acceptable risk, no token rotation needed). **GPU capacity fully measured 2026-07-12, all three GPU tiers compared (D-030-D-033)**: `find_batch_size.py` run across all 3 model tiers on RTX 4080 ($0.25/hr), RTX 5090 ($0.46/hr, 3 seq_lens), and RTX PRO 6000 ($0.91/hr, 5 seq_lens, "extreme" no-early-stop test at user's request). **Conclusion: default to RTX 5090 for all real runs — best value of the three.** RTX PRO 6000 confirmed NOT worth it (higher raw tok/s than 5090 but ~2x the price makes it the most expensive option at every tier, even pricier than the 4080 — D-033 empirically confirms D-018's VRAM-need prediction). **Self-correction, then a proper fix (D-032→D-033→D-034)**: the PRO 6000's thorough "push to real OOM" test revealed the earlier "5090 doesn't show throughput regression" claim (D-032) was based on an incomplete sweep (capped + early-stopped). The user then proposed a specific, testable hypothesis — "maybe PRO 6000 only pulls ahead at longer context" — so the 5090 was re-tested with the identical extreme methodology (D-034). **Result: the user's hypothesis was confirmed** — PRO 6000's throughput edge over the 5090 grows with sequence length (from ~2-20% at seq_len 512 to ~19-30% at seq_len 8192, across all tiers), a real memory-bandwidth-driven architectural difference. **But it doesn't flip the recommendation**: even at the widest gap (L-tier @8192), cost still favors the 5090 ($3.14 vs $4.77) since PRO 6000's ~98% price premium exceeds its largest measured speed edge (30.3%). **RTX 5090 remains the default for all real runs, now on solid ground across the full 512-8192 range tested.** All 324 raw data points (every micro_batch × tier × seq_len × GPU × methodology) saved to `docs/results/cloud_gpu_benchmarks.csv` — full narrative in `docs/learnings/20260712_gpuhub-rtx4080-capacity.md`. **Before any real run**, set `configs/train_s_*.yaml`'s `micro_batch` to the GPU-specific sweet spot (table in `docs/CLOUD_GPUHUB.md` §10, now using consistent extreme-methodology numbers for all three GPUs) — the Mac-tuned `micro_batch=16` default is suboptimal on all three cloud GPUs.

**Separately, a discussion session happened 2026-07-12** on sequence length vs. token count vs. model size — what each axis actually controls, minimum config per phase-5 learning goal (mapped onto the existing wave structure), and why the capstone's chat-context need is a deliberate separate decision. Full note: `docs/learnings/20260712_model-config-strategy.md`. Spawned **RW-5** (see Rework queue): `GPT.forward()` hard-rejects sequences longer than `max_seq_len`, blocking both Wave B's length-extrapolation probe and a wider-context L-tier capstone.

**Open item for next session: `scripts/cloud/gpuhub_setup.sh` has an uncommitted local fix** (D-029's PATH/rclone fix) that was never pushed to GitHub — this caused the exact same bug to reproduce when setting up the RTX 5090 instance via the curl-from-GitHub one-liner (worked around via `scp` instead, see D-032). Ask the user whether to commit+push this session's changes (git commits are user-initiated per CLAUDE.md — not done automatically). Projected the L-tier hero run (2.1B tokens) at ~13.7hr/~$3.43 on this tier alone — cheaper than the original 5090 "$10-20" estimate; **update `configs/train_s_*.yaml` to `micro_batch=32` before any real run on this tier** (Mac's `micro_batch=16` default isn't gpuhub's optimum). `scripts/cloud/remote.env` is now filled in for this instance so `./scripts/cloud/sync_down.sh` is one command. **Remaining:** repeat only the CUDA-version check on an actual RTX 5090 once gpuhub has inventory (everything else already proven); also flagged (not fixed) — `GPT.forward()` blocks phase 5 Wave B's length-extrapolation probe (hard-rejects seq_len > `max_seq_len`), and `find_batch_size.py`'s `mem_gb` column is unreliable (see D-030). Live playbook: `docs/CLOUD_GPUHUB.md`. | D-017 (superseded for the active path by D-027) | 4 | in-progress (essentially done pending 5090 availability) |
| RW-5 | `GPT.forward()` hard-rejects any sequence longer than `model_config.max_seq_len` (`ValueError`) — blocked (a) phase 5 Wave B's length-extrapolation probe and (b) the phase-9 capstone's chat-usability goal (real, not just extrapolated, 2k+ context). **Part (a) DONE 2026-07-12 (D-037)**: `forward()`'s guard now only applies to `learned`/`sinusoidal` (physically bounded); rope/alibi/none can run past `max_seq_len` at eval time. Probe ran clean: ALiBi improves with length (ppl 32.56→31.67 @512→2048), RoPE degrades gracefully (33.24→45.68), NoPE collapses (40→732) — real data point for part (b)'s decision, arguing ALiBi deserves consideration alongside RoPE. **Part (b) still open**: `model_l.yaml`'s `max_seq_len` (currently 512, same as S/M) should be deliberately reconsidered for the L-tier capstone per the 2026-07-12 discussion (`docs/learnings/20260712_model-config-strategy.md`) — likely ~2048 native, and now also an open question of RoPE vs ALiBi for that tier given D-037's result | Discovered incidentally while GPU-benchmarking seq_len scaling (D-030); RoPE (already the project default, D-016) is one of the position encodings best suited to this, so the fix is well-aligned with existing choices | 5 (done) / 9 (L-tier capstone max_seq_len + pos_encoding decision, still open) | in-progress |
| RW-4 | Domain corpus expansion (finance/self-help/wisdom): user picks PD-only books (Gutenberg-era finance/self-help classics — modern bestsellers are copyrighted), optionally + finance-filtered FineWeb-Edu slice; loader gets per-source mixing weights so domain share of the TRAINING STREAM (not disk) is explicit; keep domain repetition ≤~4 epochs. User's target: 10–20% (recommendation 15–25%); final % is the user's call when phase 4 builds the loader. Also: finance/wisdom probes in phase 6, domain-mix ablation in P5-G (specs updated) | User wants a finance/wisdom-steered model (2026-07-11 discussion, see `docs/learnings/20260711_gpu-vocab-datamix.md`) | 4 (loader + corpus) / 6 (probes) / 5-G (ablation) | todo |

## Parking lot (future ideas, deliberately not scheduled)

- **v2 scale-up** (after phase 9): 32k vocab + 160–180M params + ~3.2B tokens (1.6× data,
  correct Chinchilla coupling). Do NOT do mid-project: vocab change retokenizes everything and
  breaks ppl comparability with all v1 runs; 32k only pays once the corpus is big/diverse
  enough (phase 2 measured 49.3% vocab utilization at 32k on the v1 corpus). See
  `docs/learnings/20260711_gpu-vocab-datamix.md` §3.
| RW-2 | ~~Recompute D-008/D-010 if L-tier grows beyond ~105M~~ — resolved by D-015: L-tier stayed at ~105M (95.6M active), in-range of existing extrapolations, no recompute needed | D-015 finalized tier sizes vocab-aware | 3 | done |

## Run ledger (latest 10 — full list in experiments/registry.csv)

First real training runs happened this session (phases 0-3 were environment/data/tokenizer/
architecture setup, no training). Phase-4 rows so far: `20260711_p4_cpu-canary` (portability
canary, passed), `20260711_p4_s-smoke` (150 steps, val_loss 5.24), `20260711_p4_resume-test`
(bit-exact resume verified after D-023's fixes). Plus, from the still-running-as-of-session-end
overnight pipeline (D-024): `20260711_p4_s-lr-sweep-{lo,mid,hi}` and `20260711_p4_s-baseline`
(or `-auto`) — check `experiments/registry.csv`'s actual tail next session, these may not all
be present/final yet depending on when the pipeline is read. 5 non-training comparison rows
from the phase-2 tokenizer study (`20260710_p2_tokenizer-*`) are also in the registry.

## Notes for next session

- **The training engine is built** (this session): `src/llmlab/data/loader.py`
  (`MixedSourceLoader`/`Source`), `src/llmlab/train/{config,trainer}.py` (`TrainConfig`,
  `Trainer`), `scripts/train.py`, `scripts/find_batch_size.py`, plus
  `configs/train_s_{baseline,smoke,cpu_canary,lr_sweep_{lo,mid,hi}}.yaml`. See D-021 (baseline
  hyperparameters), D-022 (real MPS throughput numbers), D-023 (two resume-path bugs found and
  fixed — read this before touching `trainer.py`'s `fit()` or `Trainer.__init__`'s wandb setup
  again, the reasoning is non-obvious). `tests/test_loader.py` + `tests/test_trainer.py` are the
  reference for how the loader/trainer behave. **First check the "OVERNIGHT PIPELINE" note
  above** — `p4_s_baseline` and `p4_s_lr_sweep` may already be finished, in progress, or need a
  `--resume`/re-launch depending on when this is read.
- RW-3 status as of 2026-07-12 (this bullet supersedes older "rclone isn't installed" text):
  GitHub remote done, R2/rclone done (D-026). **Still open:** Docker Hub account + image
  build/push (Docker Desktop not installed locally), provider pod template. **Provider choice
  itself is now an open question**, not just an execution gap: the user is evaluating **gpuhub**
  as a provider; a full docs read this session (`docs/CLOUD_GPUHUB.md`) found gpuhub cannot pull
  Docker Hub images at all (conflicts with D-017's assumption, which was written RunPod-first).
  Read `docs/CLOUD_GPUHUB.md` before doing anything else on RW-3's Docker sub-step — it has the
  gpuhub-native alternative workflow (base image → setup script → Save Image) and an explicit
  "Open decision" the user needs to make (adapt to gpuhub / stay on RunPod / support both).
  RTX 5090 pricing/availability on gpuhub is also still unconfirmed (not in any of the 33 pages
  fetched) — get that page before budgeting hours. Not a blocker for S-tier engine work either
  way; walk it interactively before the first M-tier cloud run.
- RW-4 (domain corpus expansion — finance/self-help/wisdom books) still needs the user to pick
  PD-only titles; not blocking the training-engine build, but the loader's mixing-weight design
  (previous bullet) should keep RW-4 in mind so it's not a rewrite later.
- Model is ready (phase 3, D-015/D-016): `src/llmlab/model/` (`GPT`, `ModelConfig`), configs at
  `configs/model_{s,m,l}.yaml` (S 9.71M / M 34.62M / L 104.80M, deep-narrow L-tier, vocab=16000,
  head_dim=64 fixed, tied embeddings, rmsnorm/pre-norm/rope/swiglu/gpt2-init defaults, dropout
  0.0). `tests/test_model.py` is the reference for how every config axis behaves — reuse the
  `tiny_config()` pattern for training-loop unit tests rather than re-deriving fixtures.
  `notebooks/04_shapes_walkthrough.ipynb` has the tensor-shape reference if a training bug needs
  shape-by-shape debugging. Remember: `attention="mla"`, `moe`, `mtp` configs raise
  `NotImplementedError` — don't reach for them before phase 5.
- Tokenizer is decided (D-014): **HF BPE, 16,000 vocab** (corrected from an earlier "16,384"
  typo carried in the phase-3 spec — see D-015's correction note; the real tokenizer/data always
  used 16,000). Files at
  `data/tokenized/tokenizers/hf_bpe_16k/` (tokenizer itself) and
  `data/tokenized/hf_bpe_16k/{train,val}.bin` + `meta.json` (tokenized corpus, uint16 memmap,
  ready for a phase-4 DataLoader). `<|endoftext|>` id is in `meta.json`'s `eot_id` field
  (0 for this tokenizer); `<|pad|>`/`<|user|>`/`<|assistant|>` ids are already reserved in the
  vocab for phase 8 — check `data/tokenized/tokenizers/hf_bpe_16k/vocab.json` if their exact
  IDs are needed.
- Corpus is ready at `data/clean/`: `books/*.txt` (110 train + 2 val in `val/books/`),
  `dictionary_prose.txt` + `dictionary.jsonl` (+ val versions), `supplement/tinystories.txt`
  (regenerated 2026-07-11 per D-019's bug fix) + `supplement/fineweb_edu.txt` (new, D-020).
  `data/clean/manifest.json` has per-file stats/sha256/license. Re-run
  `python scripts/build_corpus.py` any time to rebuild from scratch (idempotent, cached in
  `data/raw/`); add `--force` to re-download, or `--skip-books`/`--skip-dictionary`/
  `--skip-supplement` to build a subset — partial runs merge into the existing
  `data/clean/manifest.json` rather than overwriting it.
- Token budget: 17,665,275 train + 179,655 val tokens tokenized at 16k vocab (books+dictionary,
  the S-tier ablation corpus per D-006) + TinyStories (520,469,119 tokens, 2,119,489 docs) +
  FineWeb-Edu (992,803,683 tokens, 808,365 docs) — both supplements now tokenized (D-019/D-020)
  at `data/tokenized/hf_bpe_16k/supplement_{tinystories,fineweb}.bin` with matching
  `supplement_*_docstarts.npy` doc-boundary files. Combined fresh pool ≈1.53B tokens; ~4-epoch
  Muennighoff ceiling ≈6.1B against the L-tier's ~2.1B need (D-015) — ~2.9x margin. The
  phase-4 loader (not yet built) needs per-source mixing weights to combine these four files
  by config-driven ratio into one training stream (also serves RW-4's domain-mix need later).
- Environment is ready: `source .venv/bin/activate`, `llmlab` importable, jupyter kernel `llm-lab`
  registered. `src/llmlab/utils.py` has `set_seed`, `get_device`, `param_count`, `mem_stats` —
  reuse these rather than re-deriving them in phase 3+ scripts.
- Micro-batch guidance: D-008's dummy-model bench suggested a throughput plateau around
  micro-batch 8-16 at seq_len 512, ~20.8K tok/s. **D-022 measured the real S-tier model** and
  found throughput actually flat (~11K tok/s) from micro_batch 1 through 32 — about half D-008's
  number, likely RoPE/SwiGLU/GQA's extra fixed overhead per layer. `micro_batch=16` was kept
  anyway (larger is free when flat, fewer grad-accum iterations). Re-run
  `scripts/find_batch_size.py` on any new hardware (D-018) rather than assuming either number.
- D-008 timeline tension resolved by D-010 (cloud burst option). From phase 4 onward, ALL
  training code must follow `docs/CLOUD.md` portability rules (device via
  `llmlab.utils.get_device()`/`autocast_ctx()` — already updated to be cuda>mps>cpu aware).
  The user has never rented a GPU: when the first cloud run comes up, walk CLOUD.md step by
  step and suggest the $1 practice rental first.
