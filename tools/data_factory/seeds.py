"""Seed loading — the grounding material fed into each prompt.

Three kinds:
  - "dictionary": rows of `data/clean/dictionary.jsonl` (word + pos + definitions).
  - "book_chunks": passages chunked out of `data/clean/books/*.txt`.
  - "sft_pairs": rows of an existing SFT `train.jsonl` (instruction + response) — used by
    phase 8 Part C to generate deliberately-worse "rejected" responses for DPO, grounded in the
    already-validated "chosen" response rather than the raw dictionary definition.

Every seed carries a stable `id` so batches are idempotent (re-ingesting a reply is a no-op)
and so failed seeds can be re-queued into retry batches by id.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Seed:
    """One unit of grounding material to generate examples from."""

    id: str                 # stable, content-derived
    term: str               # the headword / topic, used for grounding checks + dedup
    text: str               # the material shown to the model (definition or passage)
    raw: dict               # original record, kept for meta

    def to_prompt_block(self) -> str:
        return f"[{self.id}] {self.term}: {self.text}"


def _sid(prefix: str, key: str) -> str:
    """Short stable id: prefix + first 8 hex of sha1(key)."""
    return f"{prefix}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


def _is_real_word(word: str) -> bool:
    """A usable vocabulary headword: starts with a letter, is mostly alphabetic, ≥3 chars.
    Filters out the numerals ('1', '10'), abbreviations ('1st-class'), and chemical names
    ('1-dodecanol') that head the sorted GCIDE dump — low-value for a 'define this word'
    dataset, and models tend to silently rewrite them anyway (drifting meta.word off the seed)."""
    if len(word) < 3 or not word[0].isalpha():
        return False
    letters = sum(c.isalpha() for c in word)
    return letters >= max(3, len(word) - 1)  # allow a hyphen/apostrophe, not digits


def load_dictionary_seeds(path: str | Path, limit: int | None = None,
                          real_words_only: bool = True) -> list[Seed]:
    """Load dictionary entries as seeds. Skips entries with no usable definition, and (by
    default) non-word headwords like numerals/abbreviations/chemical formulae."""
    seeds: list[Seed] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            word = str(rec.get("word", "")).strip()
            defs = rec.get("definitions") or []
            if not word or not defs:
                continue
            if real_words_only and not _is_real_word(word):
                continue
            definition = str(defs[0]).strip()
            if len(definition) < 5:
                continue
            pos = str(rec.get("pos", "")).strip()
            term = f"{word} ({pos})" if pos else word
            seeds.append(Seed(id=_sid("dict", word), term=word, text=definition,
                              raw={"word": word, "pos": pos, "definition": definition,
                                   "display_term": term}))
            if limit and len(seeds) >= limit:
                break
    return seeds


def load_book_chunks(source: str | Path, limit: int | None = None,
                     target_chars: int = 1200) -> list[Seed]:
    """Chunk book text into ~paragraph-sized passages. `source` is a file or a directory
    of .txt files. Chunks split on blank lines and are merged up to ~`target_chars`."""
    src = Path(source)
    files = sorted(src.glob("*.txt")) if src.is_dir() else [src]
    seeds: list[Seed] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for para in paras:
            buf = f"{buf}\n\n{para}".strip() if buf else para
            if len(buf) >= target_chars:
                seeds.append(_chunk_seed(fp.stem, buf))
                buf = ""
                if limit and len(seeds) >= limit:
                    return seeds
        if buf:
            seeds.append(_chunk_seed(fp.stem, buf))
            if limit and len(seeds) >= limit:
                return seeds
    return seeds


def _chunk_seed(book: str, passage: str) -> Seed:
    return Seed(id=_sid("book", f"{book}:{passage[:64]}"), term=book, text=passage,
                raw={"book": book, "passage": passage})


def load_sft_pairs(path: str | Path, limit: int | None = None) -> list[Seed]:
    """Load an existing SFT `train.jsonl` ({instruction, response, meta.word}) as seeds whose
    grounding material is the (instruction, GOOD answer) pair itself. The id is derived from
    (word, instruction) — NOT randomized — so `load_sft_pairs` on the same file always yields the
    same ids, letting `scripts/build_dpo_pairs.py` re-derive this exact seed list later and join a
    generated rejected response back to its original chosen response without any sidecar file."""
    seeds: list[Seed] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            instruction = str(rec.get("instruction", "")).strip()
            response = str(rec.get("response", "")).strip()
            word = str((rec.get("meta") or {}).get("word", "")).strip()
            if not instruction or not response or not word:
                continue
            text = f"Instruction: {instruction}\nGood answer (do NOT reuse this — write a WORSE one): {response}"
            seeds.append(Seed(id=_sid("dpo", f"{word}:{instruction}"), term=word, text=text,
                              raw={"word": word, "instruction": instruction, "chosen": response}))
            if limit and len(seeds) >= limit:
                break
    return seeds


def select_seeds(seeds: list[Seed], exclude_ids: set[str], count: int,
                 shuffle_seed: int | None = 1337) -> list[Seed]:
    """Pick up to `count` seeds not already in `exclude_ids`.

    By default the pool is deterministically **shuffled** first (fixed seed → reproducible).
    This matters: the GCIDE dump is alphabetically sorted, so taking the first N in file order
    yields an all-'a' sample (the first real run drew 2705/2708 'a'-words before reaching 'b').
    Shuffling spreads the sample across the whole alphabet. Pass `shuffle_seed=None` to keep
    file order.
    """
    pool = [s for s in seeds if s.id not in exclude_ids]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(pool)
    return pool[:count]


def load_seeds(seed_kind: str, source: str | Path, limit: int | None = None) -> list[Seed]:
    if seed_kind == "dictionary":
        return load_dictionary_seeds(source, limit=limit)
    if seed_kind == "book_chunks":
        return load_book_chunks(source, limit=limit)
    if seed_kind == "sft_pairs":
        return load_sft_pairs(source, limit=limit)
    raise ValueError(
        f"Unknown seed_kind {seed_kind!r} (expected 'dictionary', 'book_chunks', or 'sft_pairs')"
    )
