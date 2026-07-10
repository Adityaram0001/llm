# Phase 7 — Data factory (DeepSeek-assisted dataset generation)

**Goal:** a reusable pipeline that turns "I need N examples in format X" into a validated
JSONL dataset, using DeepSeek web chat as a free human-in-the-loop generator (per D-004:
**no browser automation** — the human is the transport layer). First mission: 2–5k
instruction/Q&A pairs grounded in our books+dictionary for phase 8 SFT.
**Effort:** 1 session to build + ongoing generation batches.

## Architecture (`tools/data_factory/`)

```
tools/data_factory/
├── factory.py        # CLI: make-batches | ingest | status | export
├── tasks/            # task specs (YAML): one per dataset being built
│   └── sft_dictionary_qa.yaml
├── outbox/           # generated prompt batches → PASTE INTO DEEPSEEK (numbered .txt)
├── inbox/            # paste DeepSeek's replies here as .txt/.md (same number)
├── parsed/           # validated JSONL shards
├── failed/           # rejects + reason, auto-queued for retry batches
└── ledger.csv        # batch id, task, sent/received/valid/invalid counts, dates
```

## Task spec (YAML) — the contract

```yaml
name: sft_dictionary_qa
target_count: 3000
seed_source: data/clean/dictionary.jsonl     # or books chunks
seeds_per_prompt: 10        # items generated per pasted prompt
schema:                     # every output row must validate against this
  instruction: str          # e.g. "What does 'ephemeral' mean?"
  response: str
  meta: {word: str, style: str}
style_axes: [formal, casual, kid-friendly]   # rotated across batches for diversity
dedup_key: meta.word + style
```

## Workflow (the human loop)

1. `python factory.py make-batches --task sft_dictionary_qa --n-batches 10`
   → writes `outbox/sft_dictionary_qa_b001.txt` … each a **single self-contained prompt**:
   task instructions + K seed items + STRICT output format ("reply with ONLY a JSON array,
   schema …, no prose") + few-shot example. Sized to fit comfortably in one DeepSeek reply
   (~10–20 items per batch; long replies degrade/truncate).
2. USER: open DeepSeek chat → paste batch → copy full reply → save as `inbox/..._b001.txt`.
   New chat every few batches (fresh context = consistent quality). ~1 min per batch.
3. `python factory.py ingest` → for each inbox file: extract JSON (tolerant: strip markdown
   fences, fix trailing commas via `json-repair` approach), validate schema, dedup vs
   everything parsed so far, quality filters (length bounds, no refusals/apologies, response
   actually mentions the word, langdetect=en) → `parsed/` or `failed/`.
4. `python factory.py status` → progress vs target, failure taxonomy.
   `make-batches` automatically includes retries for failed seeds.
5. `python factory.py export --task ... --split 95/5` → `data/sft/<task>/{train,val}.jsonl`.

## Design rules (make Sonnet follow these)

- **Prompt templates live in the task YAML**, not code — new dataset = new YAML, zero code.
- Validator is paranoid: DeepSeek WILL occasionally add prose, renumber, emit smart quotes.
  Every parse failure goes to `failed/` with a reason; never silently drop.
- Batch prompts must be idempotent-safe: seeds carry ids; re-ingesting a file is a no-op.
- Optional backends behind one interface (`--backend manual|api|ollama`): DeepSeek API
  (`OPENAI`-compatible endpoint, needs key+budget approval from user) and local Ollama
  (e.g. qwen3:8b) for unattended small jobs. Same task YAML, same validator.
- Second mission later (phase 8-DPO): preference pairs — same pipeline, schema has
  `chosen`/`rejected`.

## Learning checkpoints
- Why synthetic-data diversity (style axes, seed rotation) matters — mode collapse in SFT data.
- Why strict schemas + validators beat "clean it up later".
- Cost math: what the same 3k pairs would cost via APIs (spoiler: DeepSeek API ≈ $0.05–0.3).

## Exit criteria
Factory CLI works end-to-end on a 3-batch dry run; ≥2k validated SFT pairs exported;
ledger consistent; PROGRESS/DECISIONS updated.
