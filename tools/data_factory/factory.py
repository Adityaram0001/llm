#!/usr/bin/env python
"""Data-factory CLI: make-batches | run | ingest | status | export.

Backend-agnostic, config-driven dataset generation (phase 7). A run is:

    make-batches  →  (run  OR  human paste into DeepSeek)  →  ingest  →  status  →  export

`make-batches` writes self-contained prompts to outbox/. `run` fills inbox/ automatically via
an API/local backend; for the manual backend a human does that step. `ingest` parses, validates,
dedups every inbox reply into parsed/ (rejects → failed/, never silently dropped). `export`
splits parsed/ into data/sft/<task>/{train,val}.jsonl.

Examples:
    python tools/data_factory/factory.py make-batches --task sft_dictionary_qa --n-batches 10
    python tools/data_factory/factory.py run   --task sft_dictionary_qa --backend ollama
    python tools/data_factory/factory.py ingest --task sft_dictionary_qa
    python tools/data_factory/factory.py status --task sft_dictionary_qa
    python tools/data_factory/factory.py export --task sft_dictionary_qa --split 95/5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running as a plain script (no package install) by adding the parent to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dep): KEY=VALUE lines, '#' comments, no export.
    Existing environment variables win, so an explicit shell export still overrides .env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

from data_factory.backends import get_backend           # noqa: E402
from data_factory.ledger import Ledger                   # noqa: E402
from data_factory.prompt import batch_seeds, build_prompt  # noqa: E402
from data_factory.seeds import load_seeds, select_seeds  # noqa: E402
from data_factory.spec import find_task                  # noqa: E402
from data_factory.validate import (                      # noqa: E402
    ParseError, dedup_signature, extract_json_array, validate_rows,
)

BASE = Path(__file__).resolve().parent
TASKS_DIR = BASE / "tasks"
OUTBOX, INBOX, PARSED, FAILED = BASE / "outbox", BASE / "inbox", BASE / "parsed", BASE / "failed"
LEDGER_PATH = BASE / "ledger.csv"


# --------------------------------------------------------------------------- helpers

def _parsed_path(task: str) -> Path:
    return PARSED / f"{task}.jsonl"


def _load_parsed(task: str) -> list[dict]:
    p = _parsed_path(task)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _parsed_seed_ids(rows: list[dict]) -> set[str]:
    return {r["seed_id"] for r in rows if r.get("seed_id")}


def _batch_id(task: str, idx: int) -> str:
    return f"{task}_b{idx:03d}"


def _next_batch_index(task: str) -> int:
    existing = list(OUTBOX.glob(f"{task}_b*.meta.json"))
    if not existing:
        return 1
    nums = [int(p.stem.split("_b")[-1].split(".")[0]) for p in existing]
    return max(nums) + 1


# --------------------------------------------------------------------------- commands

def cmd_make_batches(args) -> None:
    task = find_task(args.task, TASKS_DIR)
    per = args.seeds_per_prompt or task.seeds_per_prompt
    ledger = Ledger(LEDGER_PATH)

    all_seeds = load_seeds(task.seed_kind, task.seed_source)
    if args.no_shuffle and args.seed_limit:
        all_seeds = all_seeds[: args.seed_limit]  # legacy file-order dry-run cap
    done = _parsed_seed_ids(_load_parsed(task.name))
    # Shuffle by default so the sample spans the whole (alphabetically-sorted) dictionary
    # rather than starting all-'a'. Fixed seed → reproducible selection.
    shuffle_seed = None if args.no_shuffle else args.shuffle_seed
    needed = select_seeds(all_seeds, done, args.n_batches * per, shuffle_seed)
    if not needed:
        print(f"Nothing to do: all seeds already parsed for '{task.name}'.")
        return

    start_idx = _next_batch_index(task.name)
    batches = batch_seeds(needed, per)

    OUTBOX.mkdir(parents=True, exist_ok=True)
    for i, seeds in enumerate(batches):
        idx = start_idx + i
        bid = _batch_id(task.name, idx)
        style = task.style_axes[(idx - 1) % len(task.style_axes)]  # rotate axes for diversity
        prompt = build_prompt(task, seeds, style)
        (OUTBOX / f"{bid}.txt").write_text(prompt)
        (OUTBOX / f"{bid}.meta.json").write_text(json.dumps(
            {"batch_id": bid, "task": task.name, "style": style,
             "seed_ids": [s.id for s in seeds]}, indent=2))
        ledger.upsert(bid, task=task.name, style=style, backend="", n_seeds=len(seeds),
                      status="created", created=Ledger.today())
    ledger.save()

    print(f"Wrote {len(batches)} batch(es) to {OUTBOX} (b{start_idx:03d}..b{start_idx+len(batches)-1:03d}).")
    print("Next: `run --backend {api,ollama,mlx}` to auto-generate, OR paste each outbox/*.txt")
    print("      into DeepSeek web chat and save the reply to inbox/<batch_id>.txt (manual).")


def cmd_run(args) -> None:
    """Fill inbox/ for a task's outstanding batches using an automated backend."""
    find_task(args.task, TASKS_DIR)  # validate the task exists before hitting a backend
    if args.backend == "manual":
        print("manual backend is human-driven: paste outbox/*.txt into DeepSeek, save to inbox/.")
        return
    backend = get_backend(args.backend, model=args.model, temperature=args.temperature)
    ledger = Ledger(LEDGER_PATH)
    INBOX.mkdir(parents=True, exist_ok=True)

    metas = sorted(OUTBOX.glob(f"{args.task}_b*.meta.json"))
    todo = [m for m in metas if not (INBOX / f"{m.stem.replace('.meta','')}.txt").exists()]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("No outstanding batches to generate (every outbox batch already has an inbox reply).")
        return

    workers = max(1, args.workers)
    print(f"Generating {len(todo)} batch(es) via {backend.describe()} "
          f"with {workers} worker(s) ...")

    def _generate_one(meta_path: Path) -> tuple[str, bool, str]:
        """Worker: returns (batch_id, ok, detail). Runs in a thread — API calls are I/O-bound,
        so threads (not processes) give real concurrency and share the backend's usage totals."""
        bid = meta_path.stem.replace(".meta", "")
        prompt = (OUTBOX / f"{bid}.txt").read_text()
        try:
            reply = backend.generate(prompt)
        except Exception as e:  # a transient failure shouldn't kill the whole run
            return bid, False, str(e)
        (INBOX / f"{bid}.txt").write_text(reply)
        return bid, True, f"{len(reply)} chars"

    # Ledger writes happen here in the main thread as futures complete (no ledger locking needed).
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_one, m): m for m in todo}
        for fut in as_completed(futures):
            bid, ok, detail = fut.result()
            if ok:
                n_ok += 1
                ledger.upsert(bid, backend=backend.describe(), status="received",
                              received=Ledger.today())
                print(f"  {bid}: {detail}")
            else:
                n_fail += 1
                ledger.upsert(bid, backend=backend.describe(), status="gen_failed")
                print(f"  {bid}: generation FAILED ({detail})")
    ledger.save()

    totals = getattr(backend, "totals", None)
    if totals and (totals["hit"] or totals["miss"]):
        cache_pct = 100 * totals["hit"] / max(1, totals["hit"] + totals["miss"])
        print(f"Backend spend this run: ${totals['cost']:.4f} on {totals['out']:,} output tokens "
              f"(input cache-hit rate {cache_pct:.0f}%).")
    print(f"Done: {n_ok} ok, {n_fail} failed. Next: `ingest`.")


def cmd_ingest(args) -> None:
    task = find_task(args.task, TASKS_DIR)
    ledger = Ledger(LEDGER_PATH)
    parsed = _load_parsed(task.name)
    seen = {dedup_signature(r, task.dedup_paths) for r in parsed}

    PARSED.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)
    parsed_fh = _parsed_path(task.name).open("a")

    inbox_files = sorted(INBOX.glob(f"{task.name}_b*.txt"))
    total_valid = total_invalid = 0
    for reply_path in inbox_files:
        bid = reply_path.stem
        row = ledger.get(bid)
        if row and row.get("ingested") and not args.force:
            continue  # idempotent: already ingested this batch
        meta_path = OUTBOX / f"{bid}.meta.json"
        style = json.loads(meta_path.read_text())["style"] if meta_path.exists() else ""

        text = reply_path.read_text()
        try:
            rows = extract_json_array(text)
        except ParseError as e:
            (FAILED / f"{bid}.parse.txt").write_text(f"PARSE_ERROR: {e}\n\n{text}")
            ledger.upsert(bid, status="parse_failed", ingested=Ledger.today(),
                          valid=0, invalid="all")
            print(f"  {bid}: PARSE FAILED ({e})")
            total_invalid += 1
            continue

        report = validate_rows(rows, task, seen)
        for r in report.valid:
            r.setdefault("meta", {}).setdefault("style", style)
            parsed_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if report.invalid:
            with (FAILED / f"{bid}.jsonl").open("w") as ff:
                for res in report.invalid:
                    ff.write(json.dumps({"reason": res.reason, "row": res.row},
                                        ensure_ascii=False) + "\n")
        ledger.upsert(bid, status="ingested", ingested=Ledger.today(),
                      valid=len(report.valid), invalid=len(report.invalid))
        total_valid += len(report.valid)
        total_invalid += len(report.invalid)
        print(f"  {bid}: {len(report.valid)} valid, {len(report.invalid)} invalid")

    parsed_fh.close()
    ledger.save()
    print(f"Ingested: +{total_valid} valid, {total_invalid} invalid. "
          f"Total parsed now: {len(_load_parsed(task.name))} / target {task.target_count}.")


def cmd_status(args) -> None:
    task = find_task(args.task, TASKS_DIR)
    parsed = _load_parsed(task.name)
    n = len(parsed)
    pct = 100 * n / task.target_count if task.target_count else 0
    print(f"Task '{task.name}': {n} / {task.target_count} valid pairs ({pct:.1f}%)")

    # Style distribution among parsed rows (diversity check).
    styles = Counter(r.get("meta", {}).get("style", "?") for r in parsed)
    if styles:
        print("  by style:", ", ".join(f"{k}={v}" for k, v in sorted(styles.items())))

    # Failure taxonomy across failed/ shards.
    reasons: Counter = Counter()
    for ff in FAILED.glob(f"{task.name}_b*.jsonl"):
        for line in ff.read_text().splitlines():
            if line.strip():
                reasons[json.loads(line)["reason"].split("(")[0].strip()] += 1
    parse_fails = len(list(FAILED.glob(f"{task.name}_b*.parse.txt")))
    if reasons or parse_fails:
        print("  failures:")
        if parse_fails:
            print(f"    whole-batch parse failure: {parse_fails}")
        for reason, c in reasons.most_common():
            print(f"    {reason}: {c}")


def cmd_export(args) -> None:
    task = find_task(args.task, TASKS_DIR)
    parsed = _load_parsed(task.name)
    if not parsed:
        print("Nothing to export (parsed/ is empty).")
        return
    train_pct = int(args.split.split("/")[0])
    rng = random.Random(args.seed)
    rng.shuffle(parsed)
    cut = round(len(parsed) * train_pct / 100)
    train, val = parsed[:cut], parsed[cut:]

    out_dir = Path(args.out) if args.out else Path("data/sft") / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported {len(train)} train + {len(val)} val to {out_dir} (split {args.split}, seed {args.seed}).")


# --------------------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-Lab data factory")
    sub = p.add_subparsers(dest="cmd", required=True)

    mb = sub.add_parser("make-batches", help="write prompt batches to outbox/")
    mb.add_argument("--task", required=True)
    mb.add_argument("--n-batches", type=int, default=10)
    mb.add_argument("--seeds-per-prompt", type=int, default=0, help="override task default")
    mb.add_argument("--seed-limit", type=int, default=None,
                    help="with --no-shuffle only: cap the (file-order) pool, for dry runs")
    mb.add_argument("--shuffle-seed", type=int, default=1337,
                    help="deterministic seed for spreading the sample across the alphabet")
    mb.add_argument("--no-shuffle", action="store_true",
                    help="take seeds in file (alphabetical) order instead of shuffling")
    mb.set_defaults(func=cmd_make_batches)

    rn = sub.add_parser("run", help="auto-generate inbox/ replies via a backend")
    rn.add_argument("--task", required=True)
    rn.add_argument("--backend", required=True, choices=["manual", "api", "ollama", "mlx"])
    rn.add_argument("--model", default=None, help="override the backend's default model tag")
    rn.add_argument("--temperature", type=float, default=1.0)
    rn.add_argument("--limit", type=int, default=None, help="max batches this run")
    rn.add_argument("--workers", type=int, default=1,
                    help="concurrent API calls (I/O-bound thread pool; use ~8 for DeepSeek, "
                         "keep at 1 for local Ollama which is compute-bound)")
    rn.set_defaults(func=cmd_run)

    ig = sub.add_parser("ingest", help="parse+validate+dedup inbox/ into parsed/")
    ig.add_argument("--task", required=True)
    ig.add_argument("--force", action="store_true", help="re-ingest already-ingested batches")
    ig.set_defaults(func=cmd_ingest)

    st = sub.add_parser("status", help="progress vs target + failure taxonomy")
    st.add_argument("--task", required=True)
    st.set_defaults(func=cmd_status)

    ex = sub.add_parser("export", help="split parsed/ into data/sft/<task>/{train,val}.jsonl")
    ex.add_argument("--task", required=True)
    ex.add_argument("--split", default="95/5")
    ex.add_argument("--seed", type=int, default=1337)
    ex.add_argument("--out", default=None)
    ex.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> None:
    _load_dotenv(BASE.parent.parent / ".env")  # so --backend api finds DEEPSEEK_API_KEY
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
