"""Corpus acquisition: Gutenberg books, GCIDE dictionary, TinyStories supplement.

Reads the source list from `configs/corpus.yaml`, downloads raw sources into
`data/raw/`, cleans them into `data/clean/`, and returns per-file manifest
entries. Downloads are cached in `data/raw/` and skipped on re-run unless
`force=True`, so `scripts/build_corpus.py` is safe to re-run repeatedly.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import unicodedata
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# HTTP download (shared by books + GCIDE)
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, force: bool = False, timeout: int = 30) -> Path:
    """Download `url` to `dest`, skipping if it already exists (unless `force`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "llm-lab-corpus-builder/0.1"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


# ---------------------------------------------------------------------------
# Gutenberg boilerplate stripping + text cleaning
# ---------------------------------------------------------------------------

# Gutenberg headers/footers vary across decades of production; these two
# patterns cover the "*** START/END OF THE/THIS PROJECT GUTENBERG EBOOK ... ***"
# banner used since the early 2000s, which all our selected texts use.
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Cut everything outside the START/END EBOOK banners (license, headers, footers)."""
    start_match = _START_RE.search(text)
    body = text[start_match.end():] if start_match else text
    end_match = _END_RE.search(body)
    if end_match:
        body = body[: end_match.start()]
    return body.strip("\n")


_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    """Normalize unicode, collapse whitespace, and drop exact-duplicate paragraphs.

    Keeps paragraph breaks (blank lines) but collapses runs of spaces/tabs and
    3+ newlines down to a single blank line. Dedup is paragraph-hash based —
    catches repeated illustrations captions, running headers, etc. that survive
    boilerplate stripping.
    """
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)

    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    n_dropped = 0
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        kept.append(stripped)

    cleaned = "\n\n".join(kept).strip() + "\n"
    stats = {"paragraphs_kept": len(kept), "paragraphs_deduped": n_dropped}
    return cleaned, stats


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_stats(text: str) -> dict[str, int]:
    return {"chars": len(text), "words": len(text.split())}


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


def build_books(
    books: list[dict[str, Any]], raw_dir: Path, clean_dir: Path, force: bool = False
) -> list[dict[str, Any]]:
    """Download + clean each book from `configs/corpus.yaml`'s `books` list.

    Books tagged `domain: true` (RW-4's finance/self-help/wisdom set) are routed to a
    separate `domain_books/` directory rather than the general `books/` one, so
    `tokenize_corpus.py` can tokenize them into their own `.bin` and the loader can mix
    them in at an explicit, config-driven weight (Wave G's domain-mix ablation) instead of
    silently diluting the general corpus.

    Returns one manifest entry per book (source url, license, sha256, stats).
    """
    dirname = lambda is_domain: "domain_books" if is_domain else "books"  # noqa: E731
    for is_domain in (False, True):
        (clean_dir / dirname(is_domain)).mkdir(parents=True, exist_ok=True)
        (clean_dir / "val" / dirname(is_domain)).mkdir(parents=True, exist_ok=True)

    entries = []
    for book in tqdm(books, desc="books"):
        is_domain = book.get("domain", False)
        raw_path = raw_dir / "books" / f"{book['slug']}.txt"
        download_file(book["url"], raw_path, force=force)

        raw_bytes = raw_path.read_bytes()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        body = strip_gutenberg_boilerplate(raw_text)
        cleaned, clean_stats = clean_text(body)

        out_dir = clean_dir / ("val" if book.get("split") == "val" else ".") / dirname(is_domain)
        out_path = out_dir / f"{book['slug']}.txt"
        out_path.write_text(cleaned, encoding="utf-8")

        entries.append(
            {
                "type": "domain_book" if is_domain else "book",
                "slug": book["slug"],
                "title": book["title"],
                "author": book["author"],
                "source_url": book["url"],
                "gutenberg_id": book["gutenberg_id"],
                "license": "Public domain (Project Gutenberg)",
                "split": book.get("split", "train"),
                "path": str(out_path.relative_to(clean_dir)),
                "raw_sha256": sha256_bytes(raw_bytes),
                **text_stats(cleaned),
                **clean_stats,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Dictionary (GCIDE)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]*>")
# GCIDE self-closing tags (line breaks, accented-letter/typographic entities like
# <ae/>, <eacute/>, <hand/>) are written without a closing `>` — e.g. `<br/`, `<ae/` —
# so they survive `_TAG_RE` and must be swept up separately.
_UNCLOSED_TAG_RE = re.compile(r"<[a-zA-Z]+/")
_ENTRY_BLOCK_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_ENT_RE = re.compile(r"<ent>(.*?)</ent>", re.DOTALL)
_HW_RE = re.compile(r"<hw>(.*?)</hw>", re.DOTALL)
_POS_RE = re.compile(r"<pos>(.*?)</pos>", re.DOTALL)
_DEF_RE = re.compile(r"<def>(.*?)</def>", re.DOTALL)

# GCIDE letter files: CIDE.A .. CIDE.Z (no CIDE.J, .Q, .X — folded into neighbors).
_GCIDE_LETTER_FILES = [f"CIDE.{c}" for c in "ABCDEFGHIKLMNOPRSTUVWYZ"]


def _strip_tags(s: str) -> str:
    s = _TAG_RE.sub(" ", s)
    s = _UNCLOSED_TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def download_gcide(url: str, raw_dir: Path, force: bool = False) -> Path:
    """Download + extract the GCIDE tarball into `raw_dir/gcide/`."""
    archive_path = raw_dir / "gcide.tar.xz"
    extract_dir = raw_dir / "gcide"
    if extract_dir.exists() and not force:
        return extract_dir
    download_file(url, archive_path, force=force)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:xz") as tar:
        tar.extractall(extract_dir, filter="data")
    return extract_dir


def parse_gcide_entries(gcide_dir: Path) -> list[dict[str, Any]]:
    """Parse GCIDE's tagged-text letter files into {word, pos, definitions} dicts.

    GCIDE's markup is old-style SGML (unclosed tags like `<br/`), not valid
    XML, so entries are split on `<p>...</p>` blocks and fields are pulled out
    with targeted regexes rather than a real parser — simple and robust enough
    for this format.
    """
    # letter files live one directory level down, e.g. raw_dir/gcide/gcide-0.53/CIDE.A
    candidates = list(gcide_dir.glob("*/CIDE.*")) + list(gcide_dir.glob("CIDE.*"))
    letter_files = sorted({p for p in candidates if p.name in _GCIDE_LETTER_FILES})

    entries = []
    for path in tqdm(letter_files, desc="gcide letters"):
        content = path.read_text(encoding="utf-8", errors="replace")
        for block in _ENTRY_BLOCK_RE.findall(content):
            ent_match = _ENT_RE.search(block)
            if not ent_match:
                continue
            # <ent> is the clean headword; <hw> carries syllable-stress markup
            # (backticks/quotes, e.g. `Ab"sinth\``) meant for pronunciation, not prose.
            headword = _strip_tags(ent_match.group(1))
            if not headword:
                hw_match = _HW_RE.search(block)
                headword = _strip_tags(hw_match.group(1)) if hw_match else ""
            headword = re.sub(r"[`*]", "", headword).strip()
            if not headword:
                continue
            pos_match = _POS_RE.search(block)
            pos = _strip_tags(pos_match.group(1)) if pos_match else ""
            definitions = [_strip_tags(d) for d in _DEF_RE.findall(block)]
            definitions = [d for d in definitions if d]
            if not definitions:
                continue
            entries.append({"word": headword, "pos": pos, "definitions": definitions})
    return entries


def render_dictionary_prose(entries: list[dict[str, Any]]) -> str:
    """Render entries as bold-term dictionary prose, e.g.:

    **ephemeral** (adjective): lasting a very short time; transitory.
    """
    lines = []
    for e in entries:
        pos = f" ({e['pos']})" if e["pos"] else ""
        defs = "; ".join(e["definitions"])
        lines.append(f"**{e['word']}**{pos}: {defs}")
    return "\n\n".join(lines) + "\n"


def build_dictionary(
    dict_config: dict[str, Any], raw_dir: Path, clean_dir: Path, force: bool = False
) -> dict[str, Any]:
    """Download GCIDE, parse it, split off a val fraction, write prose + jsonl outputs."""
    import json
    import random

    gcide_dir = download_gcide(dict_config["url"], raw_dir, force=force)
    entries = parse_gcide_entries(gcide_dir)

    rng = random.Random(42)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * dict_config.get("val_fraction", 0.02))
    val_entries, train_entries = shuffled[:n_val], shuffled[n_val:]
    # restore original (alphabetical) order within each split for readability
    train_words = {e["word"] for e in train_entries}
    train_entries = [e for e in entries if e["word"] in train_words]
    val_words = {e["word"] for e in val_entries}
    val_entries = [e for e in entries if e["word"] in val_words]

    clean_dir.mkdir(parents=True, exist_ok=True)
    val_dir = clean_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for split, split_entries, out_dir in [("train", train_entries, clean_dir), ("val", val_entries, val_dir)]:
        prose = render_dictionary_prose(split_entries)
        prose_path = out_dir / "dictionary_prose.txt"
        prose_path.write_text(prose, encoding="utf-8")

        jsonl_path = out_dir / "dictionary.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for e in split_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        outputs[split] = {
            "prose_path": prose_path,
            "jsonl_path": jsonl_path,
            "n_entries": len(split_entries),
            **text_stats(prose),
        }

    return outputs


# ---------------------------------------------------------------------------
# Supplement (TinyStories)
# ---------------------------------------------------------------------------


def build_tinystories_supplement(clean_dir: Path, force: bool = False) -> dict[str, Any]:
    """Stream `roneneldan/TinyStories` (HF) train split to a flat text file.

    Iterates row-by-row rather than materializing the dataset as a Python
    list — the HF `datasets` library memory-maps the underlying Arrow file,
    so this stays well under the 16GB RAM ceiling regardless of dataset size.
    """
    from datasets import load_dataset

    out_dir = clean_dir / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tinystories.txt"
    if out_path.exists() and not force:
        return {"path": out_path, **text_stats(out_path.read_text(encoding="utf-8"))}

    ds = load_dataset("roneneldan/TinyStories", split="train")
    n_chars = 0
    n_words = 0
    n_stories = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="tinystories"):
            story = row["text"].strip()
            if not story:
                continue
            # Most rows contain internal blank-line paragraph breaks (the raw HF
            # text uses "\n\n" as a prose paragraph separator, not just between
            # stories) — collapse those to a single newline so "\n\n" is left
            # meaning ONLY "story boundary" for any downstream blank-line-delimited
            # reader (e.g. a streaming tokenizer). Otherwise story and paragraph
            # boundaries are indistinguishable in the flat file.
            story = re.sub(r"\n\s*\n+", "\n", story)
            f.write(story + "\n\n")
            n_chars += len(story)
            n_words += len(story.split())
            n_stories += 1

    return {"path": out_path, "chars": n_chars, "words": n_words, "n_stories": n_stories}


def build_fineweb_edu_supplement(
    clean_dir: Path, target_bytes: int, hf_config: str = "sample-10BT", force: bool = False
) -> dict[str, Any]:
    """Stream a bounded sample of `HuggingFaceFW/fineweb-edu` to a flat text file.

    Uses `streaming=True` so only the shards needed to reach `target_bytes` of raw text are
    ever fetched — the `sample-10BT` config (not the full multi-terabyte dataset) keeps that
    bounded further. Each row may itself contain internal blank-line paragraph breaks (it's
    web/edu prose, same shape as the TinyStories rows that caused D-019's bug), so those are
    collapsed to single newlines before writing — keeps `"\n\n"` meaning ONLY "document
    boundary" in the output file, same fix as `build_tinystories_supplement`.
    """
    from datasets import load_dataset

    out_dir = clean_dir / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fineweb_edu.txt"
    if out_path.exists() and not force:
        return {"path": out_path, **text_stats(out_path.read_text(encoding="utf-8"))}

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=hf_config, split="train", streaming=True)
    n_chars = 0
    n_words = 0
    n_docs = 0
    written_bytes = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="fineweb-edu", unit="doc"):
            if written_bytes >= target_bytes:
                break
            text = row["text"].strip()
            if not text:
                continue
            text = re.sub(r"\n\s*\n+", "\n", text)
            f.write(text + "\n\n")
            written_bytes += len(text.encode("utf-8")) + 2
            n_chars += len(text)
            n_words += len(text.split())
            n_docs += 1

    return {"path": out_path, "chars": n_chars, "words": n_words, "n_docs": n_docs}
