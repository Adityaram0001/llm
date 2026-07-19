"""DPO dataset: a finite set of `{instruction, chosen, rejected}` preference triples ->
paired, loss-masked (x, y) batches for direct preference optimization (phase 8, Part C).

Structurally this is `SFTDataset` (`sft_loader.py`) doubled: each triple becomes TWO independently
encoded/masked examples (one for `chosen`, one for `rejected`), sharing the same `instruction`.
DPO needs a per-sequence *summed log-probability* over the assistant span for both completions,
under both the trainable policy and the frozen reference model (see `train/dpo.py`), so the same
assistant-only mask from Part A does double duty here: it selects exactly the tokens whose
log-probs contribute to that sum.

Chosen and rejected are padded to their own batch-max width independently (their length
distributions differ — rejected responses include the deliberately-verbose failure mode) rather
than a shared width, avoiding wasted compute on whichever side is shorter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .chat_format import PAD, Message, encode_example
from .sft_loader import IGNORE_INDEX, load_jsonl


@dataclass
class DPOExample:
    chosen_ids: list[int]
    chosen_supervise: list[int]
    rejected_ids: list[int]
    rejected_supervise: list[int]
    meta: dict


def _pad_side(
    examples: list[tuple[list[int], list[int]]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same pad/shift/mask logic as `SFTDataset.collate`, factored out so chosen and rejected can
    each be padded to their own width within a batch."""
    batch_max = max(len(ids) for ids, _ in examples)
    width = batch_max - 1
    x = torch.full((len(examples), width), pad_id, dtype=torch.long)
    y = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, sup) in enumerate(examples):
        ids_t = torch.tensor(ids, dtype=torch.long)
        sup_t = torch.tensor(sup, dtype=torch.long)
        n = len(ids_t) - 1
        x[row, :n] = ids_t[:-1]
        targets = ids_t[1:].clone()
        targets[sup_t[1:] == 0] = IGNORE_INDEX
        y[row, :n] = targets
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class DPODataset:
    """Tokenizes `{instruction, chosen, rejected}` rows into paired, loss-masked batch tensors.

    `max_len` truncation follows `SFTDataset`'s convention (right-truncate, drop if nothing
    supervised remains) but is applied to chosen and rejected independently — a triple is only
    dropped if EITHER side loses all supervision, since the loss needs both.
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer: Tokenizer,
        max_len: int = 256,
        supervise_eot: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id(PAD)
        if self.pad_id is None:
            raise ValueError(f"tokenizer is missing the reserved {PAD!r} token")

        self.examples: list[DPOExample] = []
        n_truncated = n_dropped = 0
        for r in rows:
            instr = r["instruction"]
            c_ids, c_sup = encode_example(
                tokenizer, [Message("user", instr), Message("assistant", r["chosen"])],
                supervise_eot=supervise_eot,
            )
            j_ids, j_sup = encode_example(
                tokenizer, [Message("user", instr), Message("assistant", r["rejected"])],
                supervise_eot=supervise_eot,
            )
            truncated = False
            if len(c_ids) > max_len:
                c_ids, c_sup = c_ids[:max_len], c_sup[:max_len]
                truncated = True
            if len(j_ids) > max_len:
                j_ids, j_sup = j_ids[:max_len], j_sup[:max_len]
                truncated = True
            if truncated:
                n_truncated += 1
            if sum(c_sup) == 0 or sum(j_sup) == 0:
                n_dropped += 1
                continue
            self.examples.append(DPOExample(
                chosen_ids=c_ids, chosen_supervise=c_sup,
                rejected_ids=j_ids, rejected_supervise=j_sup, meta=r.get("meta", {}),
            ))

        if n_truncated:
            print(
                f"DPODataset: {n_truncated}/{len(rows)} triples had a side exceed "
                f"max_len={max_len} and were right-truncated ({n_dropped} dropped entirely)."
            )

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, tokenizer: Tokenizer, max_len: int = 256, supervise_eot: bool = True
    ) -> "DPODataset":
        return cls(load_jsonl(path), tokenizer, max_len=max_len, supervise_eot=supervise_eot)

    def collate(
        self, indices: list[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(x_chosen, y_chosen, x_rejected, y_rejected)` for the given example indices."""
        batch = [self.examples[i] for i in indices]
        x_c, y_c = _pad_side([(e.chosen_ids, e.chosen_supervise) for e in batch], self.pad_id, device)
        x_r, y_r = _pad_side(
            [(e.rejected_ids, e.rejected_supervise) for e in batch], self.pad_id, device
        )
        return x_c, y_c, x_r, y_r

    def epoch_batches(
        self, batch_size: int, seed: int, epoch: int, device: torch.device, shuffle: bool = True
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Deterministic-in-`(seed, epoch)` shuffle, mirroring `SFTDataset.epoch_batches`."""
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
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [
            self.collate(list(range(i, min(i + batch_size, len(self.examples)))), device)
            for i in range(0, len(self.examples), batch_size)
        ]
