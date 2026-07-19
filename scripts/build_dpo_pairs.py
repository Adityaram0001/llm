#!/usr/bin/env python
"""Phase 8 Part C: join generated "rejected" responses back to the existing "chosen" SFT pairs.

The data factory (`tools/data_factory/tasks/dpo_dictionary_pairs.yaml`) only generated the
REJECTED half — the CHOSEN half is the already-validated phase-7 SFT set
(`data/sft/sft_dictionary_qa/train.jsonl`), reused as-is (D-053 decision: cheaper than
regenerating both sides, and the chosen answers are already quality-gated).

The join needs no sidecar file: `seeds.load_sft_pairs` derives a stable id from (word,
instruction), so re-running it over the SAME source file reproduces the exact same ids the
factory batched against. This script re-derives that id -> (instruction, chosen) map and joins
each parsed rejected row onto it by `seed_id`.

Usage:
    python scripts/build_dpo_pairs.py
    python scripts/build_dpo_pairs.py --split 95/5 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from data_factory.seeds import load_sft_pairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chosen-source", default="data/sft/sft_dictionary_qa/train.jsonl")
    parser.add_argument(
        "--rejected-parsed", default="tools/data_factory/parsed/dpo_dictionary_pairs.jsonl"
    )
    parser.add_argument("--out", default="data/dpo/dictionary_pairs")
    parser.add_argument("--split", default="95/5")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seeds = load_sft_pairs(ROOT / args.chosen_source)
    by_id = {s.id: s.raw for s in seeds}
    print(f"re-derived {len(by_id)} (word, instruction) seeds from {args.chosen_source}")

    rejected_rows = [
        json.loads(line)
        for line in (ROOT / args.rejected_parsed).read_text().splitlines()
        if line.strip()
    ]
    print(f"loaded {len(rejected_rows)} parsed rejected rows from {args.rejected_parsed}")

    triples: list[dict] = []
    n_no_match = n_identical = 0
    for row in rejected_rows:
        raw = by_id.get(row.get("seed_id"))
        if raw is None:
            n_no_match += 1
            continue
        chosen = raw["chosen"]
        rejected = row["response"].strip()
        if rejected.strip().lower() == chosen.strip().lower():
            n_identical += 1  # a "rejected" that's word-for-word the chosen answer teaches nothing
            continue
        triples.append({
            "instruction": raw["instruction"],
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"word": raw["word"], "failure_mode": row.get("meta", {}).get("style", "?")},
        })

    print(f"joined {len(triples)} triples ({n_no_match} unmatched seed_id, {n_identical} identical-to-chosen dropped)")

    lens = sorted(len(t["rejected"]) for t in triples)
    if lens:
        p50, p99 = lens[len(lens) // 2], lens[int(len(lens) * 0.99)]
        print(f"rejected response length (chars): p50={p50} p99={p99} max={lens[-1]}")

    from collections import Counter

    print("by failure mode:", dict(Counter(t["meta"]["failure_mode"] for t in triples)))

    train_pct = int(args.split.split("/")[0])
    rng = random.Random(args.seed)
    rng.shuffle(triples)
    cut = round(len(triples) * train_pct / 100)
    train, val = triples[:cut], triples[cut:]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} train + {len(val)} val to {out_dir} (split {args.split}, seed {args.seed})")


if __name__ == "__main__":
    main()
