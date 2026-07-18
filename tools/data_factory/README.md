# Data factory (phase 7)

Config-driven, backend-agnostic pipeline that turns "I need N examples in format X" into a
validated JSONL dataset. One task YAML per dataset, one paranoid validator, swappable generators.
See `docs/phases/phase7_data_factory.md` and DECISIONS **D-048**.

## Flow

```
make-batches  →  (run  OR  human paste into DeepSeek)  →  ingest  →  status  →  export
```

```bash
# 1. write self-contained prompt batches to outbox/
python tools/data_factory/factory.py make-batches --task sft_dictionary_qa --n-batches 10

# 2a. AUTOMATED: fill inbox/ via a backend
python tools/data_factory/factory.py run --task sft_dictionary_qa --backend ollama   # local Gemma
python tools/data_factory/factory.py run --task sft_dictionary_qa --backend api      # DeepSeek
# 2b. MANUAL: paste each outbox/*.txt into DeepSeek web chat, save reply to inbox/<batch_id>.txt

# 3. parse + validate + dedup into parsed/ (rejects → failed/ with a reason)
python tools/data_factory/factory.py ingest --task sft_dictionary_qa

# 4. progress + failure taxonomy
python tools/data_factory/factory.py status --task sft_dictionary_qa

# 5. split parsed/ → data/sft/<task>/{train,val}.jsonl
python tools/data_factory/factory.py export --task sft_dictionary_qa --split 95/5
```

A new dataset = a new `tasks/<name>.yaml` (schema, seed source, styles, dedup key, prompt
instructions + few-shot). No code changes. `seed_kind: dictionary` (rows of
`data/clean/dictionary.jsonl`) or `book_chunks` (passages from `data/clean/books/`).

**Seed selection is shuffled by default** (deterministic, `--shuffle-seed 1337`): the GCIDE dump
is alphabetically sorted, so file order would sample all-'a' words (D-050 addendum). Pass
`--no-shuffle` for file order. **Batch size is backend-specific** (D-049): `--seeds-per-prompt 20`
for Gemma (bound by an ~8k output cap), `--seeds-per-prompt 60` for DeepSeek (huge output budget;
batch to cut round-trips). Concurrency: `run --workers 8` for the API (I/O-bound); keep `1` for
local Ollama (compute-bound).

## Backends

| `--backend` | What | Setup |
|---|---|---|
| `manual` | DeepSeek web chat, human transports the reply (D-004: no automation) | none |
| `api` | DeepSeek OpenAI-compatible API, `deepseek-chat` (cheap), auto prefix-caching | `DEEPSEEK_API_KEY` in `.env` |
| `ollama` | Local Gemma via Ollama daemon (default `gemma3n:e4b`) | see below |
| `mlx` | Local Gemma via Apple MLX (fastest on M-series) | see below — **separate venv** |

`--model` overrides any backend's model tag; `--temperature` is shared.

### Ollama (local Gemma) — this Mac's install is NON-brew

Homebrew on this machine is broken (`/opt/homebrew` not writable + too old for macOS 26).
Ollama was installed from its **direct macOS binary** instead — no sudo, no brew:

```bash
curl -fSL -o /tmp/Ollama-darwin.zip https://ollama.com/download/Ollama-darwin.zip
unzip -o /tmp/Ollama-darwin.zip -d "$HOME/Applications"
ln -sf "$HOME/Applications/Ollama.app/Contents/Resources/ollama" "$HOME/.local/bin/ollama"
ollama serve &                 # start the daemon (http://localhost:11434)
ollama pull gemma3n:e4b        # ~7.5 GB, one-time
```

Models live in `~/.ollama` regardless of where the binary sits. The factory hits the daemon
over HTTP, so nothing Python-side is touched.

### MLX (local Gemma) — MUST be a separate venv

`pip install mlx-lm` pulls `transformers>=5`, which conflicts with this project's pinned
`transformers<5` (the frozen phase-6 eval suite was validated on 4.x). Do **not** install it in
`.venv`. If you want the MLX backend, make an isolated env:

```bash
python -m venv .venv-mlx && source .venv-mlx/bin/activate && pip install mlx-lm
```

The MLX backend imports `mlx_lm` lazily, so the factory runs fine without it in the main venv.

## Cost (DeepSeek `deepseek-chat`)

Measured on a 36-item smoke run: **$0.0042** total → ~**$0.35 per 3000 pairs**. Prompts are
ordered invariant-prefix-first so DeepSeek's automatic context caching hits the shared
task/schema/few-shot block on every batch after the first (cheaper input tokens). `run` prints
per-batch cache-hit/miss token counts and a running cost total.

## Layout

- `spec.py` — TaskSpec + quality filters from a task YAML
- `seeds.py` — dictionary rows / book-passage chunks → id-stamped Seeds (idempotency + retry)
- `prompt.py` — one self-contained strict-JSON prompt per batch (invariant-prefix-first)
- `backends.py` — the 4 backends behind `Backend.generate()`
- `validate.py` — tolerant JSON parse + schema/quality gates + dedup (every reject keeps a reason)
- `ledger.py` — CSV audit trail, one row per batch
- `factory.py` — the CLI
- `tasks/` — task specs; `outbox/inbox/parsed/failed/` — working dirs (gitignored); `ledger.csv`
