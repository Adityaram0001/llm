"""Byte-level BPE implemented from scratch, for learning (no external tokenizer library).

Algorithm (Sennrich et al. 2016, "Neural Machine Translation of Rare Words with Subword
Units"; byte-level variant from Radford et al. 2019, GPT-2): start from the 256 possible
byte values as the base vocabulary, then repeatedly find the most frequent adjacent pair
of tokens across the (pre-tokenized) corpus and merge it into a new token, until the
target vocab size is reached. Pure Python — intended for a few MB of text (one book), not
the full corpus; see `train_hf.py` for the production tokenizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import regex as re

# GPT-2's pre-tokenizer regex (Radford et al. 2019): splits on contractions, then runs of
# letters / digits / other-symbols, then whitespace, so merges never cross these boundaries.
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str, mode: str) -> list[str]:
    """Split text into chunks that byte-pair merges are not allowed to cross.

    "none": the whole text is one chunk — merges are free to span word/space boundaries.
    "whitespace": split on the space character — merges can't cross spaces, but
      punctuation stays glued to whichever word it touches.
    "gpt2": GPT-2's regex — letters/digits/punctuation/whitespace runs kept separate.
    """
    if mode == "none":
        return [text]
    if mode == "whitespace":
        return text.split(" ")
    if mode == "gpt2":
        return GPT2_SPLIT_PATTERN.findall(text)
    raise ValueError(f"unknown pretokenize mode: {mode!r}")


def _chunk_to_byte_ids(chunk: str) -> list[int]:
    return list(chunk.encode("utf-8"))


def _count_pairs(sequences: list[list[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def _merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping left-to-right occurrence of `pair` with `new_id`."""
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


@dataclass
class ByteLevelBPE:
    """A from-scratch byte-level BPE tokenizer: call `train`, then `encode` / `decode`."""

    pretok_mode: str = "gpt2"
    merges: dict[tuple[int, int], int] = field(default_factory=dict)  # (id1, id2) -> new_id, learned order
    vocab: dict[int, bytes] = field(default_factory=lambda: {i: bytes([i]) for i in range(256)})

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> list[dict]:
        """Learn merges until `vocab_size` tokens exist.

        Returns a step-by-step log (pair merged, resulting bytes, frequency) so a notebook
        can render "the first 20 merges" or a vocab-size-vs-compression curve without
        re-running training at every intermediate size.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte base vocabulary)")
        sequences = [_chunk_to_byte_ids(c) for c in pretokenize(text, self.pretok_mode)]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        log = []
        num_merges = vocab_size - 256
        for step in range(num_merges):
            pair_counts = _count_pairs(sequences)
            if not pair_counts:
                break  # corpus exhausted: fewer distinct adjacent pairs than requested vocab
            pair, freq = pair_counts.most_common(1)[0]
            new_id = 256 + step
            sequences = [_merge_pair(seq, pair, new_id) for seq in sequences]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            log.append({"step": step, "pair": pair, "new_id": new_id, "bytes": self.vocab[new_id], "freq": freq})
            if verbose and step < 20:
                print(f"merge {step:4d}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} "
                      f"-> {self.vocab[new_id]!r}  (freq={freq})")
        return log

    def encode(self, text: str) -> list[int]:
        """Apply learned merges greedily, always preferring the earliest-learned mergeable pair."""
        id_to_pair = {new_id: pair for pair, new_id in self.merges.items()}
        ids: list[int] = []
        for chunk in pretokenize(text, self.pretok_mode):
            seq = _chunk_to_byte_ids(chunk)
            while len(seq) >= 2:
                pair_counts = _count_pairs([seq])
                mergeable_ids = [self.merges[p] for p in pair_counts if p in self.merges]
                if not mergeable_ids:
                    break
                new_id = min(mergeable_ids)
                seq = _merge_pair(seq, id_to_pair[new_id], new_id)
            ids.extend(seq)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
