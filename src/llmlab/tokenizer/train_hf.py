"""Train a production byte-level BPE tokenizer with HF `tokenizers` (GPT-2-style pipeline).

Unlike `bpe_scratch.py` (pure Python, for understanding the algorithm), this is the fast
Rust-backed implementation actually used to tokenize the full corpus in `scripts/
tokenize_corpus.py`. Special tokens reserve chat-format IDs now (`<|user|>`, `<|assistant|>`,
`<|pad|>`) even though pretraining never emits them — retrofitting them into the vocab after
phase 4/8 would shift every other token's ID and invalidate existing checkpoints.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"]


def train_byte_level_bpe(
    files: list[str],
    vocab_size: int,
    min_frequency: int = 2,
) -> ByteLevelBPETokenizer:
    """Train a GPT-2-style byte-level BPE tokenizer on the given text files."""
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    return tokenizer


def save_tokenizer(tokenizer: ByteLevelBPETokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_dir))  # vocab.json + merges.txt (human-inspectable)
    tokenizer.save(str(out_dir / "tokenizer.json"))  # single-file fast-load format


def default_corpus_files(clean_dir: Path) -> list[str]:
    """Books (train split) + dictionary prose — the S-tier corpus per DECISIONS.md D-006.

    Excludes `val/` (held out) and `supplement/` (TinyStories; separate M/L-tier stream, not
    part of the default training distribution the tokenizer should be fit to).
    """
    books = sorted(str(p) for p in (clean_dir / "books").glob("*.txt"))
    dictionary = clean_dir / "dictionary_prose.txt"
    return books + ([str(dictionary)] if dictionary.exists() else [])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tokenized/tokenizers"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8000, 16000, 32000])
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    files = default_corpus_files(args.clean_dir)
    print(f"training on {len(files)} files")

    for vocab_size in args.vocab_sizes:
        name = f"hf_bpe_{vocab_size // 1000}k"
        print(f"\n=== training {name} ===")
        tokenizer = train_byte_level_bpe(files, vocab_size=vocab_size, min_frequency=args.min_frequency)
        save_tokenizer(tokenizer, args.out_dir / name)
        print(f"saved to {args.out_dir / name}")


if __name__ == "__main__":
    main()
