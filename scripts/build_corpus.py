#!/usr/bin/env python
"""Build the phase-1 corpus: Gutenberg books + GCIDE dictionary + optional
TinyStories supplement, from `configs/corpus.yaml` into `data/clean/`.

Idempotent: downloads are cached in `data/raw/` and skipped on re-run.
Pass --force to re-download everything.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --skip-supplement
    python scripts/build_corpus.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llmlab.data import acquire  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "corpus.yaml"
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download and rebuild everything.")
    p.add_argument("--skip-books", action="store_true")
    p.add_argument("--skip-dictionary", action="store_true")
    p.add_argument("--skip-supplement", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    manifest_path = CLEAN_DIR / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    # Preserve entries for categories this run skips, so `--skip-X` runs don't
    # wipe out the other categories' entries from a prior full run.
    manifest_entries: list[dict] = []
    if args.skip_books:
        manifest_entries.extend(e for e in existing if e["type"] == "book")
    if args.skip_dictionary:
        manifest_entries.extend(e for e in existing if e["type"] == "dictionary")
    if args.skip_supplement:
        manifest_entries.extend(e for e in existing if e["type"] == "supplement")

    if not args.skip_books:
        print(f"\n=== Books ({len(config['books'])}) ===")
        manifest_entries.extend(acquire.build_books(config["books"], RAW_DIR, CLEAN_DIR, force=args.force))

    if not args.skip_dictionary:
        print("\n=== Dictionary (GCIDE) ===")
        dict_outputs = acquire.build_dictionary(config["dictionary"], RAW_DIR, CLEAN_DIR, force=args.force)
        for split, info in dict_outputs.items():
            manifest_entries.append(
                {
                    "type": "dictionary",
                    "split": split,
                    "source_url": config["dictionary"]["url"],
                    "license": config["dictionary"]["license"],
                    "path": str(info["prose_path"].relative_to(CLEAN_DIR)),
                    "n_entries": info["n_entries"],
                    "chars": info["chars"],
                    "words": info["words"],
                }
            )

    tinystories_cfg = config.get("supplement", {}).get("tinystories", {})
    if not args.skip_supplement and tinystories_cfg.get("enabled"):
        print("\n=== Supplement (TinyStories) ===")
        supp_info = acquire.build_tinystories_supplement(CLEAN_DIR, force=args.force)
        manifest_entries.append(
            {
                "type": "supplement",
                "name": "tinystories",
                "source": tinystories_cfg["hf_dataset"],
                "license": "MIT (roneneldan/TinyStories)",
                "path": str(supp_info["path"].relative_to(CLEAN_DIR)),
                "n_stories": supp_info.get("n_stories"),
                "chars": supp_info.get("chars"),
                "words": supp_info.get("words"),
            }
        )

    fineweb_cfg = config.get("supplement", {}).get("fineweb_edu", {})
    if not args.skip_supplement and fineweb_cfg.get("enabled"):
        print("\n=== Supplement (FineWeb-Edu) ===")
        fw_info = acquire.build_fineweb_edu_supplement(
            CLEAN_DIR,
            target_bytes=fineweb_cfg["target_bytes"],
            hf_config=fineweb_cfg.get("hf_config", "sample-10BT"),
            force=args.force,
        )
        manifest_entries.append(
            {
                "type": "supplement",
                "name": "fineweb_edu",
                "source": fineweb_cfg["hf_dataset"],
                "license": "ODC-BY (HuggingFaceFW/fineweb-edu)",
                "path": str(fw_info["path"].relative_to(CLEAN_DIR)),
                "n_docs": fw_info.get("n_docs"),
                "chars": fw_info.get("chars"),
                "words": fw_info.get("words"),
            }
        )

    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    totals: dict[str, dict[str, int]] = {}
    for e in manifest_entries:
        cat = e["type"]
        totals.setdefault(cat, {"chars": 0, "words": 0, "files": 0})
        totals[cat]["chars"] += e.get("chars", 0)
        totals[cat]["words"] += e.get("words", 0)
        totals[cat]["files"] += 1

    grand_chars = 0
    for cat, t in totals.items():
        est_tokens = t["chars"] // 4
        grand_chars += t["chars"]
        print(f"{cat:12s} files={t['files']:4d}  words={t['words']:>10,}  est_tokens(~chars/4)={est_tokens:>10,}")
    print(f"{'TOTAL':12s} est_tokens(~chars/4)={grand_chars // 4:>10,}")
    print(f"\nManifest written to {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
