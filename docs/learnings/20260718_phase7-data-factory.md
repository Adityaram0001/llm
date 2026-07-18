# Phase 7 — Data factory: what we built, why, the bugs, and local-vs-API throughput

Discussion session, 2026-07-18. Retrospective on building the data factory (`tools/data_factory/`)
plus a live throughput benchmark (local Gemma E4B vs DeepSeek API) and a batch-sizing analysis
driven by the two backends' very different output-side specs. Decisions: D-048 (build + backend
strategy), D-049 (this session — batch-sizing + model-deprecation + Ollama num_ctx). See also the
phase spec `docs/phases/phase7_data_factory.md`.

---

## 1. What phase 7 is, in one sentence

A config-driven, **backend-agnostic** pipeline that turns "I need N examples in format X" into a
validated JSONL dataset: one task YAML per dataset, one paranoid validator, swappable generators.
The whole point is that everything downstream of "get text from a model" (parse → validate →
dedup → export) is **identical regardless of which model produced the text**. First mission:
grounded instruction/Q&A pairs from our books+dictionary, for phase-8 SFT.

## 2. The pieces (and why each exists)

| Module | Job | Why it's separate |
|---|---|---|
| `spec.py` | `TaskSpec` + `QualityFilters` from a task YAML | New dataset = new YAML, **zero code** |
| `seeds.py` | grounding material → id-stamped `Seed`s | ids make batches idempotent + retryable |
| `prompt.py` | one self-contained strict-JSON prompt/batch | standalone prompts = manual paste works, fresh chat any time |
| `backends.py` | `manual`/`api`/`ollama`/`mlx` behind `generate()` | swap generator without touching the pipeline |
| `validate.py` | tolerant parse + schema/quality gates + dedup | the trust boundary — models WILL emit junk |
| `ledger.py` | CSV audit, one row/batch | git-diffable provenance |
| `factory.py` | CLI: make-batches / run / ingest / status / export | `run` = the automated twin of the manual paste loop |

**Key design idea that clicked:** the `run` command reads `outbox/*.txt`, calls
`backend.generate()`, writes `inbox/*.txt` — i.e. it *replaces the human* in the manual loop
while feeding the **exact same `ingest`**. So "manual DeepSeek paste" and "automated Gemma/API"
are the same pipeline with a different transport. That's what "backend-agnostic" buys you.

## 3. Decisions & why

- **Multi-backend, all four (D-048).** Manual DeepSeek web (D-004: human is the transport, no
  browser automation), DeepSeek API (fast/cheap/automated), local Gemma via **both** Ollama and
  MLX. The user's insight that *tipped* the design: our first mission is **grounded** generation
  — the answer is handed to the model in the prompt — which is a *reading-comprehension/reformat*
  task, not a knowledge task. Small local models punch far above their weight there. (Contrast:
  closed-book "invent 3k diverse finance instructions" leans on the model's own knowledge, where
  a frontier model wins clearly.)
- **E4B as the local default.** ~3GB-class effective params, comfortable on 16GB, good grounded
  quality. E2B lighter/faster; 12B better but ~7.5GB and slow.
- **No new heavy deps.** DeepSeek + Ollama are hit with plain `requests`; JSON repair is
  hand-rolled (no `json-repair`); `mlx-lm` is a lazy optional import. Keeps `requirements.txt`
  lean per CLAUDE.md.
- **Prompt ordered invariant-prefix-first (D-049).** Task instructions + schema + few-shot are
  byte-identical across every batch, so putting them as a contiguous prefix (style + seeds moved
  to the end) lets DeepSeek's automatic prefix cache hit the whole block on calls 2..N.
- **Non-thinking mode for generation.** Grounded reformatting needs NO reasoning tokens —
  `deepseek-chat` (= v4-flash non-thinking) is correct; `deepseek-reasoner` (thinking) would burn
  output budget/cost for nothing. gemma3n:e4b isn't a reasoning model either. Already optimal.

## 4. Bugs & gotchas (most only surfaced against *real* model output)

1. **Grounding gate checked the wrong field.** `require_seed_term_in_response` originally
   accepted the term appearing in the *instruction* — but the instruction echoes the word by
   construction ("What does 'X' mean?"), so the gate was meaningless. A deliberately-bad mock row
   slipped through; tightened to check the **response** only. Lesson: a quality gate must key off
   the field whose quality you actually doubt.
2. **Gemma emits U+2581 ('▁') for indentation** — the SentencePiece meta-space, NOT valid JSON
   whitespace. It broke `json.loads` on **100% of Gemma replies**. My mock replies used clean
   JSON, so tests were green while the real backend was 0% parseable. Fixed by normalizing
   `▁`→space in the tolerant parser (+regression test). Lesson: mocks that are *cleaner* than
   reality hide integration bugs; test against the real generator early.
3. **`pip install mlx-lm` silently upgraded transformers 4→5** and broke `import`, which would
   have invalidated the frozen phase-6 eval suite (validated on 4.x). Rolled back; pinned
   `transformers>=4.48,<5` + `tokenizers<0.22`; re-ran the full suite. **MLX now lives in a
   separate `.venv-mlx`.** Lesson: an ML side-tool's transitive deps can quietly rewrite your
   core env; pin upper bounds on anything the frozen work depends on.
4. **GCIDE dictionary sorts numerals/abbreviations first** ("1", "1-dodecanol", "1st-class") —
   low-value for a "define this word" dataset, and *both* models silently rewrote the headword
   (drifting `meta.word` off the seed, e.g. "1"→"first"). Added a default `real_words_only` seed
   filter (`_is_real_word`: starts-with-letter, ≥3 chars, mostly alphabetic). After it, seeds are
   real words (aardvark, abaca, …).
5. **Homebrew is broken on this Mac** — `/opt/homebrew` is owned by a *different* account
   (`MobileDev`), and the brew is too old for macOS 26.5.2. Worked around by installing Ollama
   from its direct binary (no sudo). **Trap for later:** the project venv's Python IS Homebrew's
   (`/opt/homebrew/opt/python@3.13`), so `sudo chown` + `brew update` are safe, but a blanket
   `brew upgrade` would move the python symlink and break the venv.

The validator earning its keep: on one real Gemma run it correctly rejected 3/36 numeral seeds
where the model dropped the digit from its answer — real rejects with reasons, never silent drops.

## 5. Throughput: local Gemma E4B vs DeepSeek API (the headline measurement)

Measured live this session (`scratchpad/bench_throughput.py`, 3 runs each, same 12-seed prompt):

| Metric | **Gemma E4B (Ollama, local M4)** | **DeepSeek `deepseek-chat` (API)** |
|---|---|---|
| Output speed | **~28 tok/s** (27.4 / 28.1 / 28.1) | **~139 tok/s effective** (137 / 142 / 139) |
| Per 12-item call | ~50 s | ~8 s (incl. ~2–3s network) |
| Prompt-eval | ~580 tok/s | server-side (cache hits 384→896 grew across calls) |
| Cost | $0 | ~$0.0014/call |

Ollama's numbers are **pure model speed** (`eval_count`/`eval_duration`); DeepSeek's is
**effective end-to-end** (includes network), so the real compute gap is even larger than 5×.

**Worked scale-up (measured ~96 output tokens/item):**
- 3000 pairs on Gemma: 3000 × 96 ÷ 28 ≈ **~2.9 hours** pure generation — but free, private,
  unattended.
- 3000 pairs on DeepSeek: ~288k output tokens ÷ 139 ≈ **~35 min** serial, **~$0.35 total**.

**Takeaway:** local Gemma's value is *not* speed on this hardware — it's $0 / private / no
rate-limits / no ToS friction. DeepSeek wins turnaround outright (~5× faster, ~$0.35/3k). Use
Gemma for background bulk you're not waiting on; DeepSeek when you want it now.

## 6. Batch-sizing optimization — the binding constraint differs per backend

The two backends' output-side specs are wildly different, so optimal batch size differs:

**Gemma E4B — bound by an ~8k output cap AND raw speed.**
- ~96 tok/item → 8k output ≈ 83 items *theoretical*, but JSON reliability degrades long, and
  **Ollama's default `num_ctx` (~4096) silently truncates** output past ~30 items → broken JSON →
  the whole batch fails validation. So the *effective* safe cap with default settings is ~20–25.
- Batching barely helps: the 28 tok/s output speed is the wall and it's batch-invariant; the only
  saving is amortizing the ~1085-token prompt re-encode (~1.9s at 580 tok/s) over more items.
- **Use ~20–25 items/batch.** To go bigger you must raise `num_ctx` (and accept larger retry
  blast-radius).

**DeepSeek V4-flash — effectively unbound (1M ctx / 384k output).**
- Could do 500+ items/call. But the crucial economics: **cost is output-token-dominated
  ($1.10/M out) and batch-INVARIANT** — 3000 items ≈ 288k output tokens ≈ $0.32 no matter how you
  slice them. Batching saves **wall-clock (round-trips)**, not money. Prefix caching saves *input*
  cost, which is tiny relative to output.
- Against giant batches: retry granularity (a truncated 300-item call loses 300 items) and
  coherence drift on very long structured output.
- **Use ~50–80 items/batch** — cuts 250 calls to ~40, keeps retries cheap, stays far inside limits.

**Both are usable TODAY with zero code changes** — the `--seeds-per-prompt` flag already exists:
`make-batches --seeds-per-prompt 20` (Gemma) / `--seeds-per-prompt 60` (DeepSeek).

**Model deprecation (2026-07-24):** `deepseek-chat`/`reasoner` → auto-map to `deepseek-v4-flash`
non-thinking/thinking. Names still work (backward-compat), so nothing breaks; but we should set
`DEEPSEEK_MODEL=deepseek-v4-flash` explicitly and keep non-thinking for generation.

## 7. What this session changed on paper (discussion session — no code changed)

- **D-049 logged** (batch-sizing + model-deprecation + Ollama `num_ctx` truncation risk +
  non-thinking rationale + the throughput numbers).
- **RW-7 queued** (phase 7, before the real ≥2k run): (a) set Ollama backend `num_ctx` so larger
  Gemma batches don't truncate; (b) per-backend default `seeds_per_prompt` (Gemma ~20 / DeepSeek
  ~60) so users don't have to remember the flag; (c) switch `DEEPSEEK_MODEL` to the explicit
  `deepseek-v4-flash`.

## 8. Takeaways for future me

- **Grounded vs closed-book is the axis that decides "can a small local model do this?"** If the
  answer is in the prompt, yes. If it needs world-knowledge from weights, use the frontier model.
- **Cost of API generation is output-token-dominated and batch-invariant** — batch for wall-clock
  and retry-granularity, not to save money. Cache saves input, which is the cheap side here.
- **Never trust mocks that are cleaner than the real generator** — U+2581 was invisible until a
  real Gemma call.
- **Pin upper bounds on core deps** any ML side-tool might drag forward (transformers 4→5).
- On this Mac, the honest local-vs-API tradeoff is **$0/private/unattended vs ~5×-faster/$0.35** —
  not a quality gap (E4B grounded output is genuinely close to DeepSeek's).
