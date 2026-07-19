"""SFT dataset: a finite set of `{instruction, response}` examples → padded, loss-masked
(x, y) batches for supervised fine-tuning (phase 8, Part A).

This is deliberately NOT the pretrain `MixedSourceLoader` (`loader.py`): pretraining draws random
contiguous windows from a giant memmap forever; SFT iterates a small, in-memory set of discrete
examples in shuffled epochs. The whole dictionary-QA set is ~2.9k examples of ≤~100 tokens each
(~0.37M tokens/epoch), so holding it in memory and re-shuffling per epoch is trivial.

**Padding, not packing** (D-05x): every example is padded on the right to a fixed `max_len` with
`<|pad|>`. With a causal model this is free of correction — real tokens never attend to later
pad tokens (causality), and pad *positions* carry no loss because their target is set to the
ignore index. Packing (concatenating examples to fill every window) would save compute, but at
these lengths there's almost nothing to save and it muddies the one thing this phase is here to
teach: a clean, per-token loss mask you can look at.

**The mask → targets bridge.** `GPT.forward(x, y)` computes `cross_entropy(logits, y,
ignore_index=IGNORE_INDEX)` with logits[:, i] predicting y[:, i] — targets are already the
next-token-shifted sequence (the pretrain loader does the same `window[1:]` shift). So a batch is
built from a length-(L) token sequence as `x = ids[:-1]`, `y = ids[1:]`, and then y is set to
`IGNORE_INDEX` wherever the *predicted* token isn't supervised. Since `supervise` is token-aligned
(1 on assistant-content/stop tokens), the target-frame mask is `supervise[1:]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .chat_format import PAD, Message, encode_example

# Matches GPT.forward's `cross_entropy(..., ignore_index=-1)` — masked target positions.
IGNORE_INDEX = -1


@dataclass
class SFTExample:
    """One encoded example: token-aligned `ids` and `supervise` (see `chat_format.encode_example`)."""

    ids: list[int]
    supervise: list[int]
    meta: dict


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a `{instruction, response, ...}`-per-line SFT file (data/sft/<task>/{train,val}.jsonl)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset:
    """Tokenizes `{instruction, response}` rows into padded, loss-masked (x, y) batch tensors.

    `max_len` is the padded sequence length (post-render, including markers); examples longer than
    `max_len` are right-truncated (a warning is emitted with the count, since truncating a response
    silently drops supervision). At the dictionary-QA scale nothing truncates at max_len=128.
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer: Tokenizer,
        max_len: int = 128,
        supervise_eot: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id(PAD)
        if self.pad_id is None:
            raise ValueError(f"tokenizer is missing the reserved {PAD!r} token")

        self.examples: list[SFTExample] = []
        n_truncated = 0
        n_empty_mask = 0
        for r in rows:
            msgs = [Message("user", r["instruction"]), Message("assistant", r["response"])]
            ids, sup = encode_example(tokenizer, msgs, supervise_eot=supervise_eot)
            if len(ids) > max_len:
                ids, sup = ids[:max_len], sup[:max_len]
                n_truncated += 1
            if sum(sup) == 0:  # nothing left to learn from (fully truncated response)
                n_empty_mask += 1
                continue
            self.examples.append(SFTExample(ids=ids, supervise=sup, meta=r.get("meta", {})))

        if n_truncated:
            print(
                f"SFTDataset: {n_truncated}/{len(rows)} examples exceeded max_len={max_len} and "
                f"were right-truncated ({n_empty_mask} dropped for having no supervised tokens left)."
            )

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, tokenizer: Tokenizer, max_len: int = 128, supervise_eot: bool = True
    ) -> "SFTDataset":
        return cls(load_jsonl(path), tokenizer, max_len=max_len, supervise_eot=supervise_eot)

    def collate(
        self, indices: list[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one padded, masked `(x, y)` batch from the given example indices.

        Pads each example to `batch_max` (the longest in *this* batch, ≤ max_len — dynamic padding
        keeps wasted compute low without a fixed global length). x is padded with `<|pad|>`; y is
        the next-token target with non-assistant and pad positions set to `IGNORE_INDEX`.
        """
        batch = [self.examples[i] for i in indices]
        batch_max = max(len(e.ids) for e in batch)
        # x/y are length batch_max-1 after the next-token shift.
        width = batch_max - 1
        x = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        for row, e in enumerate(batch):
            ids = torch.tensor(e.ids, dtype=torch.long)
            sup = torch.tensor(e.supervise, dtype=torch.long)
            n = len(ids) - 1  # this example's real (shifted) length
            x[row, :n] = ids[:-1]
            targets = ids[1:].clone()
            targets[sup[1:] == 0] = IGNORE_INDEX  # keep only supervised (assistant) targets
            y[row, :n] = targets
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def epoch_batches(
        self, batch_size: int, seed: int, epoch: int, device: torch.device, shuffle: bool = True
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """All (x, y) batches for one epoch. Shuffle order is deterministic in `(seed, epoch)` so a
        run is reproducible and resumable without storing sampler state (mirrors the pretrain
        loader's stateless-given-(seed, step) design)."""
        order = list(range(len(self.examples)))
        if shuffle:
            g = torch.Generator().manual_seed(seed + epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
        return [
            self.collate(order[i : i + batch_size], device)
            for i in range(0, len(order), batch_size)
        ]

    def eval_batches(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Fixed (unshuffled) batches over the whole set — for a stable masked val-loss estimate."""
        return [
            self.collate(list(range(i, min(i + batch_size, len(self.examples)))), device)
            for i in range(0, len(self.examples), batch_size)
        ]
