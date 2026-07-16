#!/usr/bin/env python
"""Tokenize the S-tier corpus (books + dictionary, per DECISIONS.md D-006) into uint16
numpy memmap files for training.

Encodes each file in `data/clean/{books,dictionary_prose.txt}` (train split) and
`data/clean/val/{books,dictionary_prose.txt}` (val split) as its own document, joined by
`<|endoftext|>`, into `data/tokenized/<name>/{train,val}.bin` + `meta.json` (vocab size,
per-document token-offset boundaries, token counts).

With `--supplement {tinystories,fineweb}`, additionally streams the matching
`data/clean/supplement/*.txt` file (one document per blank-line-separated block — both
`acquire.build_tinystories_supplement` and `acquire.build_fineweb_edu_supplement` normalize
their source text so `"\n\n"` means only "document boundary," see D-019) into
`data/tokenized/<name>/supplement_{tinystories,fineweb}.bin` (RW-1). Documents are
batch-encoded (HF tokenizers' multithreaded `encode_batch`) and appended to disk
incrementally — the raw text files (~1.9GB / ~3.6GB) are never held in memory, only one
batch of ~2000 documents at a time. Per-document doc_starts (millions of them) go to a
sibling `.npy` file rather than `meta.json`, to keep the JSON small.

Usage:
    python scripts/tokenize_corpus.py --tokenizer-dir data/tokenized/tokenizers/hf_bpe_16k
    python scripts/tokenize_corpus.py --tokenizer-dir data/tokenized/tokenizers/hf_bpe_16k --skip-corpus --supplement tinystories --supplement fineweb
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
SUPPLEMENT_DIR = CLEAN_DIR / "supplement"


def gather_split_files(clean_dir: Path, split: str) -> list[Path]:
    base = clean_dir if split == "train" else clean_dir / "val"
    books = sorted((base / "books").glob("*.txt"))
    dictionary = base / "dictionary_prose.txt"
    return books + ([dictionary] if dictionary.exists() else [])


def gather_domain_book_files(clean_dir: Path, split: str) -> list[Path]:
    base = clean_dir if split == "train" else clean_dir / "val"
    return sorted((base / "domain_books").glob("*.txt"))


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


def iter_stories(path: Path):
    """Lazily yield one story at a time from a blank-line-delimited text file.

    Uses the file object's own line iteration (buffered reads under the hood), so at most
    one story's worth of lines is held in memory regardless of file size.
    """
    buffer: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buffer:
                    story = "".join(buffer).strip()
                    buffer = []
                    if story:
                        yield story
            else:
                buffer.append(line)
    if buffer:
        story = "".join(buffer).strip()
        if story:
            yield story


def encode_supplement_streaming(
    tokenizer: Tokenizer, in_path: Path, out_path: Path, eot_id: int, batch_size: int = 2000
) -> tuple[int, np.ndarray]:
    """Batch-encode a large blank-line-delimited story file straight to disk.

    Returns (n_tokens, doc_starts). Text and token batches are both bounded to
    `batch_size` stories at a time — the ~1.9GB source file and its ~2.1M stories are
    never materialized in full.
    """
    doc_starts: list[int] = []
    n_tokens = 0
    n_docs = 0
    batch: list[str] = []
    with out_path.open("wb") as fout:

        def flush() -> None:
            nonlocal n_tokens, n_docs
            if not batch:
                return
            chunks = []
            for enc in tokenizer.encode_batch(batch):
                doc_starts.append(n_tokens)
                chunks.append(np.array(enc.ids + [eot_id], dtype=np.uint16))
                n_tokens += len(enc.ids) + 1
                n_docs += 1
            np.concatenate(chunks).tofile(fout)
            batch.clear()

        for story in tqdm(iter_stories(in_path), desc=f"encoding {in_path.name}", unit="story"):
            batch.append(story)
            if len(batch) >= batch_size:
                flush()
        flush()

    return n_tokens, np.array(doc_starts, dtype=np.int64)


def verify(tokenizer: Tokenizer, train_bin: Path, n_checks: int = 3, slice_len: int = 200) -> None:
    arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
    print(f"\n=== verifying {n_checks} random slices from {train_bin.name} ({len(arr):,} tokens) ===")
    rng = random.Random(0)
    for _ in range(n_checks):
        start = rng.randint(0, len(arr) - slice_len)
        text = tokenizer.decode(arr[start : start + slice_len].tolist())
        print(f"--- tokens[{start}:{start + slice_len}] ---")
        print(text[:300].replace("\n", " "))


SUPPLEMENT_SOURCES = {
    "tinystories": SUPPLEMENT_DIR / "tinystories.txt",
    "fineweb": SUPPLEMENT_DIR / "fineweb_edu.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="e.g. data/tokenized/tokenizers/hf_bpe_16k")
    parser.add_argument("--out-name", type=str, default=None, help="output subdir under data/tokenized/ (default: tokenizer dir name)")
    parser.add_argument("--skip-corpus", action="store_true", help="skip books+dictionary train/val (re)tokenization")
    parser.add_argument(
        "--supplement",
        action="append",
        choices=sorted(SUPPLEMENT_SOURCES),
        default=[],
        help="additionally tokenize a supplement source (RW-1); repeatable",
    )
    parser.add_argument(
        "--domain-books",
        action="store_true",
        help="additionally tokenize data/clean/{domain_books,val/domain_books} (RW-4) into "
        "domain_books.bin / domain_books_val.bin, one file = one document, same convention "
        "as the main corpus",
    )
    parser.add_argument(
        "--books-only",
        action="store_true",
        help="additionally tokenize data/clean/{books,val/books} WITHOUT the dictionary into "
        "books_only.bin / books_only_val.bin -- Wave G's multi-epoch overfitting lab wants a "
        "books-only pool of known size to control epoch count precisely",
    )
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

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"tokenizer_dir": str(args.tokenizer_dir), "vocab_size": vocab_size, "eot_id": eot_id})
    meta.setdefault("splits", {})
    meta.setdefault("supplements", {})

    if not args.skip_corpus:
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
        verify(tokenizer, out_dir / "train.bin")

    if args.books_only:
        meta.setdefault("books_only", {})
        for split in ["train", "val"]:
            base = CLEAN_DIR if split == "train" else CLEAN_DIR / "val"
            files = sorted((base / "books").glob("*.txt"))
            print(f"=== books_only {split}: {len(files)} files ===")
            ids, doc_starts = encode_split(tokenizer, files, eot_id)
            suffix = "" if split == "train" else "_val"
            out_path = out_dir / f"books_only{suffix}.bin"
            docstarts_path = out_dir / f"books_only{suffix}_docstarts.npy"
            ids.tofile(out_path)
            np.save(docstarts_path, np.array(doc_starts, dtype=np.int64))
            meta["books_only"][split] = {
                "n_tokens": int(len(ids)),
                "n_docs": len(files),
                "path": str(out_path.relative_to(ROOT)),
                "doc_starts_path": str(docstarts_path.relative_to(ROOT)),
            }
            print(f"  {len(ids):,} tokens -> {out_path}")
        verify(tokenizer, out_dir / "books_only.bin")

    if args.domain_books:
        meta.setdefault("domain_books", {})
        for split in ["train", "val"]:
            files = gather_domain_book_files(CLEAN_DIR, split)
            if not files:
                raise FileNotFoundError(f"no domain_books files found for split={split}")
            print(f"=== domain_books {split}: {len(files)} files ===")
            ids, doc_starts = encode_split(tokenizer, files, eot_id)
            suffix = "" if split == "train" else "_val"
            out_path = out_dir / f"domain_books{suffix}.bin"
            docstarts_path = out_dir / f"domain_books{suffix}_docstarts.npy"
            ids.tofile(out_path)
            np.save(docstarts_path, np.array(doc_starts, dtype=np.int64))
            meta["domain_books"][split] = {
                "n_tokens": int(len(ids)),
                "n_docs": len(files),
                "path": str(out_path.relative_to(ROOT)),
                "doc_starts_path": str(docstarts_path.relative_to(ROOT)),
            }
            print(f"  {len(ids):,} tokens -> {out_path}")
        verify(tokenizer, out_dir / "domain_books.bin")

    for name in args.supplement:
        in_path = SUPPLEMENT_SOURCES[name]
        if not in_path.exists():
            raise FileNotFoundError(f"supplement source not found: {in_path}")
        print(f"\n=== supplement: {name} ({in_path}) ===")
        out_path = out_dir / f"supplement_{name}.bin"
        docstarts_path = out_dir / f"supplement_{name}_docstarts.npy"
        n_tokens, doc_starts = encode_supplement_streaming(tokenizer, in_path, out_path, eot_id)
        np.save(docstarts_path, doc_starts)
        meta["supplements"][name] = {
            "n_tokens": int(n_tokens),
            "n_docs": int(len(doc_starts)),
            "path": str(out_path.relative_to(ROOT)),
            "doc_starts_path": str(docstarts_path.relative_to(ROOT)),
        }
        print(f"  {n_tokens:,} tokens, {len(doc_starts):,} docs -> {out_path}")
        verify(tokenizer, out_path)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nmeta written to {meta_path}")


if __name__ == "__main__":
    main()
