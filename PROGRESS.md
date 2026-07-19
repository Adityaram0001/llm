# PROGRESS — single source of truth for project state

> Every Claude session reads this first and updates it last. Keep it honest and terse.
> Status values: `todo` | `in-progress` | `done` | `blocked` | `skipped`

**Active phase:** Phases 0-7 are **done** (milestones M1/M2/M3 declared). **Phase 8
(`docs/phases/phase8_finetuning.md`, fine-tuning: SFT/LoRA/DPO) is IN PROGRESS — Parts A (SFT,
D-051) and B (LoRA from scratch, D-052) are DONE (2026-07-19)**; Part C (DPO) remains, so
**milestone M4 is NOT yet declared** (it lands when the whole phase is complete). Wave G's deferred
dictionary-ablation item (D-045: "does the dictionary in the mix improve a 'define X' eval?")
remains unblocked but unrun (parked, see parking lot).

**This session, Part B (2026-07-19, LoRA from scratch — D-052):** built LoRA end-to-end (no peft):
`src/llmlab/train/lora.py` (`LoRALinear` = frozen base + `(α/r)·BA`, B=0-init, apply/merge/state,
attn/attn+ffn/ffn presets), `SFTConfig.lora` + LoRA branch in `SFTTrainer` (adapter-only optimizer
+ checkpoints), `src/llmlab/train/sft_infer.py` `load_finetuned` (reconstructs base+adapter; eval/
chat refactored onto it), `scripts/compare_finetune.py`, 3 configs, `tests/test_lora.py` (+10,
suite **183 pass**). Ran the rank+placement sweep (r8/r32 attn, r8 attn+ffn). **Result: LoRA is
13–53× cheaper in AdamW optimizer memory (116.6MB→2.2–8.9MB) and competitive-to-better in quality —
r8 attn+ffn (val 3.777) beat the full FT (3.828); placement > rank.** Adapters are 0.77–2.98MB vs
full FT's 116.7MB ckpt. Two real bugs caught by running (bf16-autocast dtype clash → `F.linear`;
CPU/MPS device placement → `model.to(device)`) that the fp32/cpu tests missed; 8 stale registry
rows from the crashed pre-fix launches removed by hand. Report: `docs/results/finetune_report.md`
Part B. **NEXT: Part C (DPO), then the multi-base SFT addendum (below).**

**Planned addendum (user-requested, after Part C):** SFT-on-many-bases — now that each SFT is
~2 min, fine-tune several pretrained checkpoints to test **"does better pretrain loss predict a
better SFT model?"**. All ~55 S-tier checkpoints are already LOCAL (no R2 pull). Cleanest set (same
`model_s.yaml` arch, training-only differences, so directly comparable + drop-in loadable):
`p4_s_baseline` (val 3.50), `wave-d-muon` (~3.35, biggest pretrain win), `wave-d-wsd` (~3.38). The
scaling runs (5/10/25/50M) are a second axis ("does a bigger base SFT better?") but need their own
model configs. Architecture-changing runs (qk-norm/ALiBi/MLA/MoE) each need their matching config
to load. Not in the phase-8 spec — a bonus ablation; `scripts/compare_finetune.py` already handles
the tabulation.

**This session (2026-07-19, phase 8 Part A — SFT):** built the whole from-scratch SFT stack (no
trl/peft) and ran the project's first supervised fine-tune. New code: `src/llmlab/data/
{chat_format,sft_loader}.py` (chat-ML template with the phase-2-reserved specials + an
**assistant-only loss mask**, exact-by-construction so it dodges RW-6's boundary fragility),
`src/llmlab/train/{sft_config,sft_trainer}.py` (`SFTTrainer`: warm-start from a pretrain ckpt,
fresh AdamW, epoch loop, masked loss, **a live catastrophic-forgetting probe**), `scripts/
{sft,eval_sft,chat}.py`, `configs/sft_s_dictionary.yaml`, `tests/test_sft.py` (+15 tests, full
suite **173 pass**). Run `20260719_p8_sft-s-dictionary` (base `p4_s_baseline`, lr 2e-5, 3 epochs,
~1m44s on the M4). **Result (D-051): a behavior flip, not a knowledge gain.** SFT val loss
5.54→3.83; instruction **stop-rate 0%→80%** (base continues in book-prose forever, SFT answers &
emits `<|endoftext|>`), answer length 64→34 tok. Dict MC accuracy barely moved (26.5%→29.5%, both
near 25% chance), cloze 0% — a 10M base has no real definitional knowledge for SFT to surface; SFT
teaches the protocol, not the content. **Catastrophic forgetting measured live: pretrain-val ppl
34.93→40.10 (+14.8%).** The chat REPL (`scripts/chat.py`) works — the payoff moment. Eval quoted
only RW-6-safe metrics (`definition_completion_ppl` skipped). Report: `docs/results/
finetune_report.md`. **NEXT: Part B (LoRA from scratch) then Part C (DPO).**

**This session (2026-07-18, phase 7 — data factory BUILD):** built the whole backend-agnostic
generation pipeline in `tools/data_factory/` (`spec/seeds/prompt/backends/validate/ledger/
factory.py` + `tasks/sft_dictionary_qa.yaml`). Resolved the spec's "optional backends behind one
interface" decision point with the user up front: **all 4 backends built** — `manual` (DeepSeek
web, D-004 human transport), `api` (DeepSeek OpenAI-compatible, needs a key), `local` via **both**
`ollama` (`gemma3n:e4b` default) **and** `mlx` (`mlx-community/gemma-3n-E4B-it-4bit`, lazy
import). The user added the **local Gemma** option (E2B/E4B/12B on the 16GB M4) on the correct
insight that our grounded first mission is a reading-comprehension/reformat task where a small
local model does well; **E4B chosen as the default local size.** No new deps (used `requests`;
in-house tolerant JSON repair instead of `json-repair`; `mlx-lm` optional/Apple-only). CLI:
`make-batches | run | ingest | status | export`; `run` is the automated analogue of the manual
paste loop (fills `inbox/` from `outbox/`) so the SAME paranoid validator serves every backend.
**3-batch dry run passed** (mock replies with injected fences/smart-quotes/trailing-commas/prose/
refusal/grounding-fail/duplicate rows): tolerant parse recovered all, bad rows all rejected with
reasons, re-ingest idempotent (+0), export split clean, ledger consistent. **15 new tests
(`tests/test_data_factory.py`), full suite 154 passed (was 139).** Dry-run artifacts cleaned up
(no fake pairs in repo); `parsed/`/`failed/`/`ledger.csv`/`data/sft/` gitignored. Full story +
the "grounding gate must check the response, not the instruction" bug caught during the dry run:
D-048.

**Same session, later (2026-07-18, backends stood up + tested for real — D-048 execution
notes):** user gave a DeepSeek API key ($2 budget) + Gemma go-ahead. **Both backends now live and
validated:** DeepSeek `api` (36/36 valid, **~$0.35/3k** on the cheap `deepseek-chat` model,
prefix-caching wired by reordering the prompt invariant-first) and local **Gemma `gemma3n:e4b`
via Ollama** (quality genuinely good on grounded dictionary Q&A, ~52s/batch, free/unattended).
**Homebrew is BROKEN on this Mac** (`/opt/homebrew` not writable + too old for macOS 26) — Ollama
was installed from its direct binary instead (no sudo; `~/Applications/Ollama.app`). **MLX
deferred to a separate venv:** `pip install mlx-lm` upgraded transformers 4→5 and would have
broken the frozen eval suite — rolled back, pinned `transformers<5`/`tokenizers<0.22`. Two real
robustness fixes that only real model output surfaced (mocks missed them): Gemma emits U+2581
('▁') for indentation → broke JSON parse → normalized in the tolerant parser; and GCIDE sorts
numerals/abbrevs first → added a default `real_words_only` seed filter. New `tools/data_factory/
README.md`. Suite **156 pass**. **NEXT: generate ≥2k pairs at scale** — both backends proven,
just a volume run (user picks backend/count).

**⚠️ Known bug, not yet fixed (RW-6, D-047, found 2026-07-17 in a discussion session):
`dictionary_probes.py`'s `definition_completion_ppl` is computed on silently corrupted text
(100% of examples affected) — do not quote that one number from any existing `eval_results.json`
as real until RW-6 is fixed. Every other metric in the suite (val ppl/bpb, MC accuracy, cloze,
domain probes, HellaSwag, LAMBADA-style, generation) is verified unaffected. Full story:
`docs/learnings/20260717_phase6-eval-suite-deep-dive.md`.**

**This session (2026-07-16/17, phase 6 — evaluation suite):** built the full core eval battery:
`src/llmlab/eval/{scoring,perplexity,dictionary_probes,domain_probes,generation,benchmarks,
report}.py` + `scripts/evaluate.py`. Three decision points resolved with the user up front
(AskUserQuestion, all recommended options chosen): HellaSwag uses the REAL public validation set
(`Rowan/hellaswag` on the HF Hub, subsampled 200/run) rather than a hand-written toy set; domain
probes (finance/wisdom, RW-4) are 24 hand-written items (`data/eval/domain_probes.json`) since
phase 7's data factory doesn't exist yet to generate them; and the eval_deep_dive notebook's
early/mid/final trio comes from a genuinely fresh milestone-checkpoint run
(`20260717_p6_s-p6-baseline-milestones`, new `TrainConfig.milestone_steps`, same seed/recipe as
`p4_s_baseline`, run on the already-idle RTX 5090) rather than mixing checkpoints from different
runs. That milestone run's final val_loss (3.4954) reproduced the original baseline (3.5037)
within the D-035 noise floor — a nice bonus reproducibility check. `notebooks/
08_eval_deep_dive.ipynb` (executes cleanly) finds the spec's predicted "smooth ppl vs flat
accuracy" split in real numbers, plus a live calibration/reliability-diagram analysis
(ECE=0.0164, well-calibrated despite low absolute accuracy) and a benchmark-contamination
discussion. Caught two real bugs before finishing: a `.view()`/stride crash in the new
perplexity module (fixed with `.contiguous()`), and a test that was silently appending fake rows
to the REAL `experiments/registry.csv` (Trainer's `REGISTRY_PATH` is a module-level constant,
not `run_dir`-relative — fixed via `monkeypatch`, 3 spurious rows removed by hand). Full story:
D-046. 139 tests pass (was 127). 1 new run registered. **GPU LEFT RUNNING (singapore-b:25864,
confirmed live at session start, used for ~4 minutes this session) — user must stop it to stop
billing** once satisfied everything synced down correctly (checkpoints already pulled and
verified loadable locally).

**Earlier (2026-07-16, Wave G — data & scaling, LAST wave of phase 5):** resolved RW-4
(open since phase 1): curated **62 public-domain finance/self-help/wisdom-practical books**
(Adam Smith, Ricardo, Bagehot, Keynes, Ford, Taylor, Samuel Smiles, James Allen, Orison Swett
Marden, Russell Conwell, Barnum, Hubbard, Ruskin, Booker T. Washington, etc.) from local
Gutenberg catalog data, hand-vetted out of a noisy auto-categorized list (excluded fiction,
off-theme academic psychology, one low-quality modern author). New general-purpose pipeline
code: `domain`-tagged book routing (`acquire.build_books`), `--domain-books`/`--books-only`
tokenize modes — no loader changes needed (`MixedSourceLoader`'s per-source weights already
supported this). Ran 11 S-tier runs on the RTX 5090 (singapore-b:25864): **domain-mix ablation
(4 runs, D-045)** found a strictly monotonic general-val-loss cost as finance/wisdom share rises
(0%→10%→25%→50%: 3.980→4.015→4.055→4.144) — recommend 10-25% share for the capstone, not 50%.
**Multi-epoch overfitting lab (3 runs)** confirmed the train/val gap opens as predicted
(+0.268→+0.344→+0.921 at 1/4/16 epochs) even though val_loss itself plateaus rather than
worsening at this scale. **Mini scaling law (4 runs, 5/10/25/50M params @ fixed 200M tokens)**
found the real headline result: comparing each run's BEST (early-stopped) vs FINAL val_loss
shows 5M/10M still improving at the end, but **25M and especially 50M overfit the repeated
17.66M-token pool well before the budget ends** (50M peaks at step 1650 of 3050, then worsens
+0.109 while train_loss keeps falling) — bigger models overfit a small repeated pool FASTER, a
clean tie to the project's own Muennighoff-ceiling concept (D-015/RW-1). Fit
`L(N)=11909.67·N^-0.694+3.102` on best values (final values would have mis-ranked 25M vs 50M).
`notebooks/07_scaling_law.ipynb` (executes cleanly) reproduces every number. **Wrote
`docs/results/recipe.md`** (deferred since Wave C) consolidating all 7 waves' winning choices
into phase 9's starting config, with an explicit "still open" section. Figure:
`docs/results/wave_g_data_scaling.png`. 11 runs registered with real verdicts, +0 new tests
needed (no new model/trainer code, only data-pipeline additions). **GPU LEFT RUNNING
(singapore-b:25864) — user must stop it to stop billing** once satisfied everything synced down
correctly.

**Earlier (2026-07-16, Wave F — DeepSeek specials, MoE + MTP):** implemented
`src/llmlab/model/moe.py` (`MoEFFN` — 8 fine-grained routed experts + 1 shared, top-2, expert
hidden sized so active params/token match the dense baseline) and `src/llmlab/model/mtp.py`
(`MTPHead` — sequential Multi-Token-Prediction depths sharing the main output head), wired
through `Block`/`GPT`/`Trainer` with no remaining `NotImplementedError` guards on `moe`/`mtp`
config fields. +34 tests (127 local cpu/mps, 98 remote-cuda, all pass). Ran 3 S-tier runs on the
RTX 5090 (singapore-b:25864, live and idle at session start). **Results (D-044):
DeepSeekMoE reproduces its headline win** — both balancing methods beat the dense control by
~0.09 val_loss (>4x noise floor) at matched active params (18.61M total vs control's 9.71M, both
~4.43M active) — more total capacity via fine-grained experts genuinely helps. **aux_loss vs
bias_free balancing are statistically tied on final quality** (0.008 apart) but **bias_free
balances measurably slower** (aux_loss's gradient signal reaches good balance by step ~200;
bias_free's bounded per-step bias nudge takes until step ~800-1000) — a clean reproduction of
DeepSeek-V3's own mechanistic tradeoff. **MTP (+1 head predicting t+2) is not distinguishable
from noise** (+0.017, at the noise floor's edge) at this scale/token budget, though the extra
head does demonstrably learn its own harder task. **Caught and fixed a real bug before writing
any verdict**: `Trainer.evaluate()` was reading `forward()`'s COMBINED training loss (main CE +
weighted aux terms) instead of pure CE, which silently added ~+0.15 to the aux_loss run's
val_loss (moe_aux_loss sums across all 15 layers) while bias_free's was unaffected (zero aux
loss by design) — producing a fake ~0.15 "gap" that looked like a real finding. Caught by
checking the delta against the D-035 noise floor before concluding anything; fixed
(`last_aux_metrics["ce_loss"]` now separates pure CE from the training objective), covered by a
new regression test, both affected runs re-executed clean (the two buggy, notes-less, same-
session run folders were deleted rather than kept as confusing duplicates). Figure:
`docs/results/wave_f_deepseek_specials.png`. **GPU LEFT RUNNING (singapore-b:25864) — user must
stop it to stop billing** once satisfied everything synced down correctly.

**Later same day (2026-07-16, wandb turned on + a real cloud-throughput gap fixed):** user
created a wandb account and gave credentials — stored in `.env`/`.env.example` (D-042), new
`docs/WANDB.md` written for future sessions to reference, added to README's map + quick-start.
**Caught a real bug before declaring success**: the user-given `WANDB_ENTITY=adityaram0001` is
invalid (that's the username, not the entity) — the first sync attempt printed "done." for all
33 offline runs but every one had actually failed server-side (verified via the pod's debug
log, not just the CLI's own output). Corrected the entity by querying `wandb.Api().viewer`
directly (real entity: `adityaram0001-bbiq-technologies-private-limited`), re-ran
`scripts/cloud/wandb_sync.sh` (new script), and this time independently verified via
`api.runs(...)`: **33/33 runs present, all finished, spot-checked values match known results.**
Added `--wandb-online` to `scripts/train.py` (mirrors `--device`) so any future cloud run can
stream live with one flag, without flipping D-009's offline-by-default for every other run.
**Also found and fixed a real efficiency gap while doing this**: grepped every wave config's
`micro_batch` and confirmed **all 12 of Waves A/B/C's S-tier cloud runs used the Mac-tuned
`micro_batch=16` instead of the 5090's measured `mb=64` sweet spot** — a documented warning in
`docs/CLOUD_GPUHUB.md` (written before Wave A even ran) got missed because later waves' configs
were copy-pasted from earlier ones. Quality verdicts are unaffected (loss is factorization-
invariant, D-040's own finding), only wall-clock/GPU-hours were left on the table. Wave D
onward already self-corrected to `mb=64`. Fixed properly this time with a **runtime warning**
(`Trainer.__init__`, `src/llmlab/train/trainer.py`) that fires whenever `device=="cuda"` and
`micro_batch<=16`, since the doc alone already proved insufficient (D-043). **Read the runtime
warning if it ever prints — it means a config needs fixing before spending real GPU-hours.**

**This session (2026-07-16, checkpoint archival to R2):** user flagged that 40 of 48 runs'
checkpoints existed ONLY on the gpuhub pod's data disk (7.2GB, growing every wave, never backed
up) — built `scripts/cloud/archive_checkpoints.py` + `scripts/cloud/push_checkpoints.sh`
(D-041): strips optimizer state from `ckpt/best.pt` before archiving (model weights only, ~39MB
vs ~111MB full — ablation runs are reproducible from config+seed, not meant to be resumed),
except named fork-point runs (currently just `wave_d_constant`, which two Wave D WSD runs
really `--resume`d from) which archive full+resumable. Pushed **server→R2 directly**, no Mac
round-trip. First run archived all 48 runs, 1.395 GiB in ~110s; R2 bucket `llm` now totals
4.274 GiB (2.879 GiB data + 1.395 GiB experiments) — comfortably inside the free 10GB tier, well
under the user-approved 50GB ceiling. Nothing deleted from the pod (user's call — data disk has
37GB free, not under pressure yet); re-run `push_checkpoints.sh` any time, it only pushes
new/changed files. Also answered two side questions: **git pull cannot delete checkpoints or
wandb logs** (both gitignored/untracked on every machine, git only touches tracked files) — but
found the pod's git tree is stale (HEAD at Wave D's `3d330cc`, uncommitted drift in
`registry.csv`/`PROGRESS.md`/model files from the trainer's own local writes) and will likely
refuse a plain `git pull` until that drift is discarded, not fixed this session, flagged in
D-041. **wandb comparison across runs has never been set up** (project has run offline-only per
D-009 since the start) — phase 6 does NOT cover this (it's the eval-metrics suite, not a
training-curve dashboard); `notebooks/05_compare_runs.ipynb` already does cross-run comparison
locally without it. Left as optional/not-blocking. **GPU still up in no-GPU/cheap mode
(singapore-b:25864, $0.10/hr) — user stops it when done, not part of this session's scope.**

**Earlier (2026-07-13, Wave E):** implemented the efficiency/memory knobs Wave E needed
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
| 5 | Ablation lab (research techniques) | `docs/phases/phase5_ablations.md` | done |
| 6 | Evaluation suite | `docs/phases/phase6_evaluation.md` | done |
| 7 | Data factory (DeepSeek-assisted) | `docs/phases/phase7_data_factory.md` | done |
| 8 | Fine-tuning: SFT / LoRA / DPO | `docs/phases/phase8_finetuning.md` | in-progress (Part A/SFT done, B/C todo) |
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
- [x] Wave F — DeepSeek specials (D-044): new `src/llmlab/model/moe.py` (`MoEFFN`, 8 routed + 1
  shared experts, top-2, active-param-matched) + `src/llmlab/model/mtp.py` (`MTPHead`, sequential
  depths, shared output head), fully wired through `Block`/`GPT`/`Trainer` (no more
  `NotImplementedError` guards). **DeepSeekMoE reproduces its headline win** — both balancing
  methods beat the dense control by ~0.09 val_loss (>4x noise floor) at matched active params
  (18.61M total vs 9.71M control, ~4.43M active either way). **aux_loss vs bias_free balancing
  tied on final quality** (0.008 apart) but **bias_free balances measurably slower** (aux_loss's
  gradient signal reaches good balance by step ~200, bias_free's bounded per-step nudge takes
  until step ~800-1000) — a clean reproduction of DeepSeek-V3's own tradeoff. **MTP (+1 head
  predicting t+2) not distinguishable from noise** (+0.017) at this scale/budget, though the
  extra head demonstrably learns its own harder task. Real val_loss-measurement bug caught and
  fixed before any verdict was written (moe_aux_loss was silently leaking into the eval metric —
  see D-044); covered by a new regression test. 3 runs registered, +34 tests (127 local, 98
  remote-cuda). Figure: `docs/results/wave_f_deepseek_specials.png`.
- [x] Wave G — Data & scaling (D-045), the LAST wave of phase 5. **Multi-epoch overfitting lab**
  (3 runs, books-only 14.14M-token pool, 1/4/16 epochs): train/val gap opens as predicted
  (+0.268→+0.344→+0.921) though val_loss itself plateaus rather than worsening at this scale.
  **RW-4 domain-mix ablation** (4 runs, 62 newly-curated finance/self-help/wisdom books, fixed
  49.15M-token budget): strictly monotonic general-val-loss cost as domain share rises
  (0%→10%→25%→50%: 3.980→4.015→4.055→4.144) — recommend 10-25% share for the capstone.
  **Mini scaling law** (4 runs, 5/10/25/50M params @ fixed 200M tokens,
  `notebooks/07_scaling_law.ipynb`): bigger models (25M, especially 50M) overfit the repeated
  17.66M-token pool well before the budget ends (50M peaks at step 1650/3050, then worsens
  +0.109) — a real, size-dependent tie to the project's Muennighoff-ceiling concept; fit on
  best/early-stopped val_loss, `L(N)=11909.67·N^-0.694+3.102`. **Dictionary ablation explicitly
  deferred to phase 6** (needs eval probes that don't exist yet — the spec itself allows this).
  Figure: `docs/results/wave_g_data_scaling.png`. 11 runs registered.
- [x] Exit criteria (M2): waves A-D done + verdicts, figures in `docs/results/`, all runs
  registered. **M2 DECLARED 2026-07-13.**
- [x] `docs/results/recipe.md` written 2026-07-16 (deferred since Wave C until Wave G landed) —
  consolidates every wave's winning choices into phase 9's L-tier hero-run starting config.
  **Phase 5 is now fully DONE.**

## Phase 6 checklist (done)

- [x] `src/llmlab/eval/` + `scripts/evaluate.py --ckpt <path> [--suite core]` — writes
  `eval_results.json` into the checkpoint's run folder; reads model config + tokenizer from that
  run folder's own `config.yaml` (no extra CLI flags needed for a normal run). Registry-column
  update explicitly skipped (D-046: schema-stability risk outweighs the convenience, revisit if
  cross-run eval comparison becomes a frequent need).
- [x] Core suite, all sub-items: val ppl + bits-per-byte on books/dictionary SEPARATELY
  (`perplexity.py`, new `--dictionary-only` tokenize flag → `dictionary_only_val.bin`); 3
  dictionary probes (`dictionary_probes.py`: definition-completion ppl, 4-way MC-by-loglik,
  cloze); domain probes (`domain_probes.py` + `data/eval/domain_probes.json`, 24 hand-written
  finance/proverb/advice items, RW-4); 15-prompt generation battery + distinct-n/seq-rep-4
  (`generation.py`, temp 0.8/top-p 0.95 frozen per the spec's decision point); HellaSwag (real
  `Rowan/hellaswag` validation set, 200/run) + homemade LAMBADA-style last-word accuracy
  (`benchmarks.py`).
- [x] `notebooks/08_eval_deep_dive.ipynb` — runs the suite live on 3 fresh milestone checkpoints
  (early/mid/final, new run `20260717_p6_s-p6-baseline-milestones`, D-046); calibration/
  reliability-diagram analysis (ECE=0.0164); benchmark-contamination discussion. Executes
  cleanly end to end. Figures: `docs/results/phase6_{eval_deep_dive,calibration}.png`.
- [x] Decision points: eval battery contents FROZEN after this phase (D-046); generation
  sampling params (temp 0.8/top-p 0.95) confirmed as-specified.
- [x] Exit criteria: `evaluate.py --suite core` runs in **~38s** on the baseline checkpoint
  (spec: <10min); results JSON schema stable; **M3 DECLARED 2026-07-17.**
- [x] New `TrainConfig.milestone_steps` (Trainer/config.py) — named checkpoint snapshots at
  specific step counts, alongside (never replacing) latest.pt/best.pt. +12 tests
  (`tests/test_eval.py` 11, `tests/test_trainer.py` +1) — 139 total, all pass.

## Phase 7 checklist (in-progress)

- [x] Backend strategy decided with the user (D-048): all 4 backends built —
  `manual` / `api` (DeepSeek) / `local` via BOTH `ollama` + `mlx` (Gemma, E4B default).
- [x] `tools/data_factory/` architecture built: `spec.py` (TaskSpec + QualityFilters),
  `seeds.py` (dictionary rows OR book-passage chunks, id-stamped for idempotency+retry),
  `prompt.py` (self-contained strict-JSON prompt, style rotated per batch), `backends.py`
  (4 backends behind `Backend.generate()`), `validate.py` (tolerant parse + schema + quality
  gates + dedup, every reject keeps a reason), `ledger.py` (CSV audit), `factory.py`
  (`make-batches | run | ingest | status | export`).
- [x] First task spec: `tasks/sft_dictionary_qa.yaml` (3000 grounded dictionary Q&A,
  styles formal/casual/kid-friendly, dedup on word+style).
- [x] 3-batch dry run end-to-end (mock replies with injected messiness) — parse/validate/dedup/
  idempotency/export all verified. +tests (`tests/test_data_factory.py`).
- [x] **Backends stood up + validated on REAL output (D-048 execution notes, 2026-07-18):**
  DeepSeek `api` (36/36 valid, ~$0.35/3k, cheap model + prefix-caching wired) AND local Gemma
  `ollama` (`gemma3n:e4b`, quality genuinely good, ~52s/batch). MLX deferred to a separate venv
  (transformers-5 conflict). Two real fixes surfaced by real output: Gemma's U+2581 meta-space
  (broke JSON parse — fixed) + a `real_words_only` seed filter (GCIDE sorts numerals/abbrevs
  first). Full suite **156 pass**.
- [x] **Generated SFT pairs (D-050, 2026-07-18)** via DeepSeek **deepseek-v4-flash non-thinking**,
  60 seeds/batch, `run --workers 8` concurrency (~3–4 min/50 batches), ~$0.08/run.
- [x] **Caught + fixed an alphabetical-skew bug (D-050 addendum):** the first run was 2705/2708
  **'a'-words** (sorted GCIDE read in file order). Fix: `seeds.select_seeds` deterministic shuffle,
  now the `make-batches` default (`--no-shuffle` opts out). **Regenerated diversified: 3,070 pairs**
  spread across the whole alphabet (a:214…s:353…p:276), styles 1002/991/957, 0 dedup collisions.
  The a-words set is **archived** at `data/sft/sft_dictionary_qa_a-words/` (user asked to keep it).
- [x] **Exported** the diversified set to `data/sft/sft_dictionary_qa/{train,val}.jsonl`
  (2916/154, 95/5, seed 1337), gitignored — **push to R2 before any cloud phase-8 SFT run**.
- [x] **Exit criteria met → PHASE 7 DONE (2026-07-18):** CLI end-to-end ✓, ≥2k pairs ✓ (3070),
  ledger consistent ✓, PROGRESS/DECISIONS updated ✓.

## Phase 8 checklist (in-progress)

**Part A — SFT (DONE 2026-07-19, D-051):**
- [x] Chat format: `src/llmlab/data/chat_format.py` (render/encode with reserved specials
  `<|user|>`/`<|assistant|>`/`<|endoftext|>`/`<|pad|>`; `add_generation_prompt` for inference).
- [x] `src/llmlab/data/sft_loader.py`: tokenize `data/sft/*/train.jsonl` → right-pad → **assistant-only
  loss mask** (non-assistant + pad targets = -1, honored by the model's existing `ignore_index=-1`).
  Pad-not-pack (examples are tiny); mask exact-by-construction (dodges RW-6's boundary fragility).
- [x] `src/llmlab/train/{sft_config,sft_trainer.py}` + `scripts/sft.py`: warm-start from
  `p4_s_baseline`, lr 2e-5, 3 epochs, bf16; tracks **frozen pretrain-val ppl** alongside SFT loss
  (catastrophic forgetting, live). Run `20260719_p8_sft-s-dictionary`, ~1m44s on the M4.
- [x] Eval before/after (`scripts/eval_sft.py`, `eval_sft.json`): stop-rate 0%→80%, answer len
  64→34, dict MC 26.5%→29.5%, **pretrain ppl 34.93→40.10 (+14.8% forgetting)**. RW-6-safe metrics
  only (`definition_completion_ppl` skipped). Table: `docs/results/finetune_report.md`.
- [x] `scripts/chat.py` — minimal REPL (single- and multi-turn, one-shot pipe mode). Works. 🎉
- [x] +15 tests (`tests/test_sft.py`), full suite **173 pass**.
- [ ] (deferred nice-to-have) notebook visualizing a masked example + the forgetting curve.

**Part B — LoRA from scratch (DONE 2026-07-19, D-052):**
- [x] `src/llmlab/train/lora.py`: `LoRALinear` (frozen `W` + `(α/r)·BA`, A~kaiming/**B=0** so init==base),
  `apply_lora`/`merge_lora`/`lora_state_dict`/`load_lora_state`, target presets (attn/attn+ffn/ffn,
  never `lm_head` — tied). +10 tests.
- [x] `SFTConfig.lora` + `SFTTrainer` LoRA branch (freeze base, AdamW over adapters only, **adapter-only
  checkpoints**, tok/s logging); `src/llmlab/train/sft_infer.py` `load_finetuned` (reconstructs
  base+adapter; `eval_sft.py`+`chat.py` refactored onto it).
- [x] Rank+placement sweep: `configs/sft_s_dictionary_lora_{r8_attn,r32_attn,r8_attnffn}.yaml`, 3 runs.
- [x] `scripts/compare_finetune.py` → the full-FT-vs-LoRA table. **LoRA 13–53× cheaper optimizer
  memory (116.6→2.2–8.9MB), adapters 0.77–2.98MB vs 116.7MB; quality competitive-to-better (r8
  attn+ffn 3.777 beat full FT 3.828); placement > rank; stop-rate 90–95% vs 80%.** Forgetting
  higher on the adapted model (+24–37%) but LR-confounded AND fully reversible (frozen base intact).
- [x] 2 real bugs caught by running (bf16-autocast dtype → `F.linear`; CPU/MPS device → `.to(device)`);
  8 stale registry rows from crashed pre-fix launches removed. Report: `docs/results/finetune_report.md`.

**Part C — DPO (todo):** preference pairs via the data factory (chosen vs deliberately-worse
rejected), `src/llmlab/train/dpo.py` (policy vs frozen ref, β·log-ratio), train from the SFT ckpt,
track reward margins & KL drift, eval battery again.

**Exit criteria (M4, not yet met):** chat REPL demo ✓ (Part A), before/after table ✓ (Parts A+B),
LoRA ✓ (Part B), DPO done, all runs registered, decisions logged. M4 declared when C lands.

## Rework queue (see CLAUDE.md "Change management")

| ID | What | Why | Fix in phase | Status |
|----|------|-----|--------------|--------|
| RW-1 | Tokenize TinyStories supplement + a FineWeb-Edu sample with hf_bpe_16k → `data/tokenized/hf_bpe_16k/supplement_*.bin`. **Fully done 2026-07-12**: tokenized (D-019, D-020: 520.5M + 992.8M tokens) AND pushed to R2 (D-026) — `r2:llm/data/tokenized/` now has all 16 files (train/val, both supplements + docstarts, meta.json, all 3 tokenizer vocabs), 2.879 GiB, verified via `rclone lsf -R` | D-015: L-tier is 105M, needs ~2.1B tokens; repetition alone (~4 epochs of core+TinyStories) was right at the edge, so a FineWeb-Edu sample was added for margin + topic diversity | 4 | done |
| RW-3 | One-time cloud accounts setup. **Done:** GitHub remote, R2 bucket + rclone (D-026), Docker Desktop installed locally, $10 gpuhub credit purchased, provider decision (D-027: gpuhub, native image-snapshot workflow, RunPod kept documented-but-unbuilt). **Cloud pipeline validated live end-to-end 2026-07-12 (D-029)**: RTX 4080 Super dry-run instance (D-028), `scripts/cloud/gpuhub_setup.sh` ran clean over SSH, real training smoke test passed (99,554 tok/s), checkpoint round-tripped CUDA→Mac-MPS. Image saved as **"genesis"** — contains OS+deps+conda env+our SSH key+`.env` (system disk only; repo/data live on the data disk, NOT in the image — user confirmed the `.env`-in-image finding is an acceptable risk, no token rotation needed). **GPU capacity fully measured 2026-07-12, all three GPU tiers compared (D-030-D-033)**: `find_batch_size.py` run across all 3 model tiers on RTX 4080 ($0.25/hr), RTX 5090 ($0.46/hr, 3 seq_lens), and RTX PRO 6000 ($0.91/hr, 5 seq_lens, "extreme" no-early-stop test at user's request). **Conclusion: default to RTX 5090 for all real runs — best value of the three.** RTX PRO 6000 confirmed NOT worth it (higher raw tok/s than 5090 but ~2x the price makes it the most expensive option at every tier, even pricier than the 4080 — D-033 empirically confirms D-018's VRAM-need prediction). **Self-correction, then a proper fix (D-032→D-033→D-034)**: the PRO 6000's thorough "push to real OOM" test revealed the earlier "5090 doesn't show throughput regression" claim (D-032) was based on an incomplete sweep (capped + early-stopped). The user then proposed a specific, testable hypothesis — "maybe PRO 6000 only pulls ahead at longer context" — so the 5090 was re-tested with the identical extreme methodology (D-034). **Result: the user's hypothesis was confirmed** — PRO 6000's throughput edge over the 5090 grows with sequence length (from ~2-20% at seq_len 512 to ~19-30% at seq_len 8192, across all tiers), a real memory-bandwidth-driven architectural difference. **But it doesn't flip the recommendation**: even at the widest gap (L-tier @8192), cost still favors the 5090 ($3.14 vs $4.77) since PRO 6000's ~98% price premium exceeds its largest measured speed edge (30.3%). **RTX 5090 remains the default for all real runs, now on solid ground across the full 512-8192 range tested.** All 324 raw data points (every micro_batch × tier × seq_len × GPU × methodology) saved to `docs/results/cloud_gpu_benchmarks.csv` — full narrative in `docs/learnings/20260712_gpuhub-rtx4080-capacity.md`. **Before any real run**, set `configs/train_s_*.yaml`'s `micro_batch` to the GPU-specific sweet spot (table in `docs/CLOUD_GPUHUB.md` §10, now using consistent extreme-methodology numbers for all three GPUs) — the Mac-tuned `micro_batch=16` default is suboptimal on all three cloud GPUs.

**Separately, a discussion session happened 2026-07-12** on sequence length vs. token count vs. model size — what each axis actually controls, minimum config per phase-5 learning goal (mapped onto the existing wave structure), and why the capstone's chat-context need is a deliberate separate decision. Full note: `docs/learnings/20260712_model-config-strategy.md`. Spawned **RW-5** (see Rework queue): `GPT.forward()` hard-rejects sequences longer than `max_seq_len`, blocking both Wave B's length-extrapolation probe and a wider-context L-tier capstone.

**Open item for next session: `scripts/cloud/gpuhub_setup.sh` has an uncommitted local fix** (D-029's PATH/rclone fix) that was never pushed to GitHub — this caused the exact same bug to reproduce when setting up the RTX 5090 instance via the curl-from-GitHub one-liner (worked around via `scp` instead, see D-032). Ask the user whether to commit+push this session's changes (git commits are user-initiated per CLAUDE.md — not done automatically). Projected the L-tier hero run (2.1B tokens) at ~13.7hr/~$3.43 on this tier alone — cheaper than the original 5090 "$10-20" estimate; **update `configs/train_s_*.yaml` to `micro_batch=32` before any real run on this tier** (Mac's `micro_batch=16` default isn't gpuhub's optimum). `scripts/cloud/remote.env` is now filled in for this instance so `./scripts/cloud/sync_down.sh` is one command. **Remaining:** repeat only the CUDA-version check on an actual RTX 5090 once gpuhub has inventory (everything else already proven); also flagged (not fixed) — `GPT.forward()` blocks phase 5 Wave B's length-extrapolation probe (hard-rejects seq_len > `max_seq_len`), and `find_batch_size.py`'s `mem_gb` column is unreliable (see D-030). Live playbook: `docs/CLOUD_GPUHUB.md`. | D-017 (superseded for the active path by D-027) | 4 | in-progress (essentially done pending 5090 availability) |
| RW-5 | `GPT.forward()` hard-rejects any sequence longer than `model_config.max_seq_len` (`ValueError`) — blocked (a) phase 5 Wave B's length-extrapolation probe and (b) the phase-9 capstone's chat-usability goal (real, not just extrapolated, 2k+ context). **Part (a) DONE 2026-07-12 (D-037)**: `forward()`'s guard now only applies to `learned`/`sinusoidal` (physically bounded); rope/alibi/none can run past `max_seq_len` at eval time. Probe ran clean: ALiBi improves with length (ppl 32.56→31.67 @512→2048), RoPE degrades gracefully (33.24→45.68), NoPE collapses (40→732) — real data point for part (b)'s decision, arguing ALiBi deserves consideration alongside RoPE. **Part (b) still open**: `model_l.yaml`'s `max_seq_len` (currently 512, same as S/M) should be deliberately reconsidered for the L-tier capstone per the 2026-07-12 discussion (`docs/learnings/20260712_model-config-strategy.md`) — likely ~2048 native, and now also an open question of RoPE vs ALiBi for that tier given D-037's result | Discovered incidentally while GPU-benchmarking seq_len scaling (D-030); RoPE (already the project default, D-016) is one of the position encodings best suited to this, so the fix is well-aligned with existing choices | 5 (done) / 9 (L-tier capstone max_seq_len + pos_encoding decision, still open) | in-progress |
| RW-6 | `src/llmlab/eval/scoring.py`'s `encode_prompt_continuation` silently assumes the separately-encoded prompt is a token-prefix of the jointly-encoded (prompt+continuation) sequence — false whenever a BPE merge pulls the prompt's trailing character into a token that belongs to the continuation. **Verified exhaustively (2026-07-17, discussion session, D-047): 3,281/3,281 (100%) of `dictionary_probes.py`'s definition-completion examples hit this** (prompt always ends `": "`) — the continuation gets silently corrupted (a real word, e.g. "excessive", dropped entirely), so `definition_completion_ppl` in EVERY `eval_results.json` written so far (baseline + all 3 milestone checkpoints) is computed on mangled text and must not be quoted as-is. Cloze/domain-probes/HellaSwag/LAMBADA-style are all verified NOT affected (different, safe prompt/continuation boundary shape). A second, smaller issue in the same audit: the MC probe doesn't use this helper at all and tokenizes each choice standalone (~10% get a different, out-of-context token count — doesn't corrupt content, just adds noise). **Fix path spelled out in D-047**: split by character offset (`Tokenizer.encode(...).offsets`) instead of by separately-encoded token count. Not fixed yet — discussion sessions change no code. **Phase 8 Part A (2026-07-19, D-051) USED the eval suite but did not MODIFY it, so RW-6 was skirted not fixed: `scripts/eval_sft.py` quotes only the RW-6-safe metrics (MC accuracy, cloze) and explicitly omits `definition_completion_ppl`.** Still open; fix when `src/llmlab/eval/` is next *modified* | Found while writing the phase-6 learnings note, by checking the helper's own docstring claim against the real tokenizer instead of trusting it | 6 (whenever `src/llmlab/eval/` is next touched) | todo |
| RW-4 | Domain corpus expansion (finance/self-help/wisdom). **Corpus + ablation DONE 2026-07-16 (D-045)**: 62 PD books curated from local Gutenberg catalog data (finance/investing/economics/business + self-help/personal-development/wisdom-practical categories), `domain`-tagged routing added to `acquire.build_books`/`scripts/tokenize_corpus.py`, `MixedSourceLoader`'s existing per-source weights used unmodified. Domain-mix ablation (0/10/25/50% share) found a strictly monotonic general-val-loss cost — recommend **10-25% share** for the capstone (not the original 10-20% target's upper bound, and well short of 50%). **Eval probes DONE 2026-07-17 (D-046)**: `src/llmlab/eval/domain_probes.py` + 24 hand-written items (`data/eval/domain_probes.json`) — this baseline (0% domain share) scores exactly at chance on them, the expected null result. **Still open:** the Wave G "does dictionary-in-the-mix help a define-X eval" ablation is now unblocked but not yet RUN; growing the 6.76M-token domain pool before L-tier's much bigger token budget is an open question (recipe.md flags it) | User wants a finance/wisdom-steered model (2026-07-11 discussion, see `docs/learnings/20260711_gpu-vocab-datamix.md`) | 9 (capstone domain-share decision) | in-progress (corpus+ablation+probes done, capstone decision open) |
| RW-7 | Data-factory generation optimizations, to apply BEFORE the real ≥2k run (D-049): (a) set the Ollama backend's `options.num_ctx` high enough (≥8192) that larger Gemma batches don't **silently truncate** output past ~4096 default ctx → broken JSON → whole-batch failure; (b) give per-backend default `seeds_per_prompt` (Gemma ~20–25, DeepSeek ~50–80) so users needn't remember `--seeds-per-prompt` (the flag already works today); (c) set `DEEPSEEK_MODEL=deepseek-v4-flash` explicitly (the `deepseek-chat` alias deprecates 2026-07-24 but auto-maps, so non-urgent), keeping NON-thinking mode for grounded generation | Throughput benchmark + spec review this session (D-049): Gemma bound by 8k output cap + Ollama's silent num_ctx truncation; DeepSeek cost is output-token-dominated + batch-invariant so batch for wall-clock not money | 7 (before the ≥2k generation run) | **(c)+DeepSeek max_tokens DONE (D-050)**; (a) Ollama num_ctx + (b) per-backend seeds_per_prompt defaults still open — only needed before a large-batch LOCAL Gemma run (the flag works today) |

## Parking lot (future ideas, deliberately not scheduled)

- **Clean LoRA rank sweep** (spawned by the 2026-07-19 discussion session): our r8-vs-r32 sweep held
  **α=16 fixed**, so the scaling `α/r` differed (2.0 vs 0.5) — not a pure rank comparison. A clean
  rerun should hold `α/r` constant (α ∝ r). Doesn't change any Part-B conclusion (r8≈r32 was a tie;
  placement dominated). See `docs/learnings/20260719_phase8-sft-lora-deep-dive.md` §5.
- **LR-matched full-FT-vs-LoRA forgetting** (same discussion; also flagged in D-052): our LoRA runs
  used lr 5e-4 vs full FT's 2e-5, so the adapted-model forgetting (+24–37% vs +15%) is LR-confounded.
  Rerun at a matched LR for a fair comparison. The *reversibility* conclusion (frozen base intact)
  holds regardless.
- **v2 scale-up** (after phase 9): 32k vocab + 160–180M params + ~3.2B tokens (1.6× data,
  correct Chinchilla coupling). Do NOT do mid-project: vocab change retokenizes everything and
  breaks ppl comparability with all v1 runs; 32k only pays once the corpus is big/diverse
  enough (phase 2 measured 49.3% vocab utilization at 32k on the v1 corpus). See
  `docs/learnings/20260711_gpu-vocab-datamix.md` §3.
- **Wave F MoE equal-wall-clock rerun**: Wave F's -0.09 val_loss win for DeepSeekMoE (D-044) is
  at fixed TOKEN budget; MoE also measured ~2.18x slower tok/s than the dense control (median
  223.6K vs 487.9K tok/s, routing's many-small-matmuls overhead — found in the 2026-07-16
  discussion session, not in the original D-044 write-up). Whether MoE still wins at fixed
  WALL-CLOCK is untested — would need the dense control trained ~2.18x longer (or MoE trained
  ~2.18x fewer steps) at the same GPU-time budget. Worth resolving before phase 9 commits to MoE
  for the capstone if training wall-clock (not just token count) is a real constraint. See
  `docs/learnings/20260716_wave-f-deepseek-specials.md` §9.
- **Wave G dictionary ablation (deferred, phase-5 spec explicitly allows this):** "with vs
  without dictionary in the mix → does a 'define X' eval improve?" — **unblocked 2026-07-17**:
  phase 6's dictionary probes (`src/llmlab/eval/dictionary_probes.py`) now exist. Still a cheap,
  not-yet-run ablation (single training-data-mix axis, reuses the existing S-tier corpus, no new
  model code, just 2 short train runs + `scripts/evaluate.py` on each).
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

- **Phase 7 factory is BUILT + BOTH backends live/validated (2026-07-18, D-048 + execution
  notes); the remaining work is GENERATION at scale.** Setup is DONE: DeepSeek key is in `.env`
  (cheap `deepseek-chat`, ~$0.35/3k, $2 budget), Ollama is installed from its **direct binary**
  (NOT brew — brew is broken here: `/opt/homebrew` not writable + too old for macOS 26; the
  binary is `~/Applications/Ollama.app/Contents/Resources/ollama`, symlinked at `~/.local/bin/
  ollama`) with `gemma3n:e4b` pulled. **To generate the ≥2k pairs:** start the Ollama daemon
  (`ollama serve &`) if using local, then
  `python tools/data_factory/factory.py make-batches --task sft_dictionary_qa --n-batches N`,
  then `run --backend ollama` (free/unattended, ~52s/batch) OR `run --backend api` (fast, cheap,
  prints cost) OR the manual DeepSeek-web loop; then `ingest` → `status` → `export --split 95/5`
  → `data/sft/sft_dictionary_qa/{train,val}.jsonl` for phase 8. `--model`/`--temperature`
  override per run. **MLX needs its own `.venv-mlx`** (transformers-5 conflict — do NOT
  `pip install mlx-lm` into `.venv`; see `tools/data_factory/README.md`). Reference: the README,
  `tasks/sft_dictionary_qa.yaml`, and `tests/test_data_factory.py` (validator behavior).
  A `book_chunks` seed_kind is already supported for book-grounded Q&A (new task YAML, zero code).
- **Phase 6 is fully done (2026-07-17), M3 declared.** Next session should read `docs/phases/
  phase7_data_factory.md` and start phase 7 (DeepSeek-assisted data factory). Remember
  CLAUDE.md's D-004 rule: never build/run browser automation against DeepSeek's web UI —
  human-in-the-loop batch workflow only, or the API backend if the user enables it.
- **The eval suite is built and frozen** (phase 6, D-046): `src/llmlab/eval/` (`scoring.py`'s
  `score_continuation`/`mc_by_loglik`/`greedy_continuation`/`encode_prompt_continuation` are the
  shared primitives every probe reuses — reach for these instead of writing new log-likelihood
  code), `scripts/evaluate.py --ckpt <path>` (writes `eval_results.json` into the run folder,
  reads model/tokenizer from that run's own `config.yaml`). `tests/test_eval.py` is the
  reference for probe behavior (uses a tiny GPT with the REAL 16k tokenizer — vocab size must
  match, unlike `test_model.py`'s fully-toy 256-vocab fixtures). Do NOT add/change probe content
  without a new D-entry — the battery is frozen so future phase-8/9 checkpoints stay comparable
  to phase 6/9's own numbers. `TrainConfig.milestone_steps` (new this phase) is available any
  time a run needs named intermediate snapshots, not just latest/best.
  `docs/results/recipe.md` is still the input for eventually assembling the phase-9 hero config
  — read it before phase 9, not before phase 7.
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
