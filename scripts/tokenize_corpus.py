#!/usr/bin/env python
"""Tokenize the S-tier corpus (books + dictionary, per DECISIONS.md D-006) into uint16
numpy memmap files for training.

Encodes each file in `data/clean/{books,dictionary_prose.txt}` (train split) and
`data/clean/val/{books,dictionary_prose.txt}` (val split) as its own document, joined by
`<|endoftext|>`, into `data/tokenized/<name>/{train,val}.bin` + `meta.json` (vocab size,
per-document token-offset boundaries, token counts).

Does NOT include the TinyStories supplement — that stays a separate, un-tokenized stream
until an M/L-tier run actually mixes it in (see PROGRESS.md notes for phase 2).

Usage:
    python scripts/tokenize_corpus.py --tokenizer-dir data/tokenized/tokenizers/hf_bpe_16k
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"


def gather_split_files(clean_dir: Path, split: str) -> list[Path]:
    base = clean_dir if split == "train" else clean_dir / "val"
    books = sorted((base / "books").glob("*.txt"))
    dictionary = base / "dictionary_prose.txt"
    return books + ([dictionary] if dictionary.exists() else [])


def encode_split(tokenizer: Tokenizer, files: list[Path], eot_id: int) -> tuple[np.ndarray, list[int]]:
    """Encode each file as its own document, joined by an end-of-text token.

    Returns the flat token array plus each document's start offset (for a phase-4
    dataloader that wants to avoid sampling windows that straddle unrelated documents).
    """
    all_ids: list[int] = []
    doc_starts: list[int] = []
    for f in files:
        doc_starts.append(len(all_ids))
        ids = tokenizer.encode(f.read_text(encoding="utf-8")).ids
        all_ids.extend(ids)
        all_ids.append(eot_id)
    return np.array(all_ids, dtype=np.uint16), doc_starts


def verify(tokenizer: Tokenizer, train_bin: Path, n_checks: int = 3, slice_len: int = 200) -> None:
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    print(f"\n=== verifying {n_checks} random slices from {train_bin.name} ({len(arr):,} tokens) ===")
    rng = random.Random(0)
    for _ in range(n_checks):
        start = rng.randint(0, len(arr) - slice_len)
        text = tokenizer.decode(arr[start : start + slice_len].tolist())
        print(f"--- tokens[{start}:{start + slice_len}] ---")
        print(text[:300].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="e.g. data/tokenized/tokenizers/hf_bpe_16k")
    parser.add_argument("--out-name", type=str, default=None, help="output subdir under data/tokenized/ (default: tokenizer dir name)")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError(f"vocab_size={vocab_size} doesn't fit in uint16 (max 65535)")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise ValueError("tokenizer is missing the <|endoftext|> special token")

    out_dir = ROOT / "data" / "tokenized" / (args.out_name or args.tokenizer_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"tokenizer_dir": str(args.tokenizer_dir), "vocab_size": vocab_size, "eot_id": eot_id, "splits": {}}
    for split in ["train", "val"]:
        files = gather_split_files(CLEAN_DIR, split)
        print(f"=== {split}: {len(files)} files ===")
        ids, doc_starts = encode_split(tokenizer, files, eot_id)
        out_path = out_dir / f"{split}.bin"
        ids.tofile(out_path)
        meta["splits"][split] = {
            "n_tokens": int(len(ids)),
            "n_docs": len(files),
            "doc_starts": doc_starts,
            "path": str(out_path.relative_to(ROOT)),
        }
        print(f"  {len(ids):,} tokens -> {out_path}")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nmeta written to {out_dir / 'meta.json'}")

    verify(tokenizer, out_dir / "train.bin")


if __name__ == "__main__":
    main()
