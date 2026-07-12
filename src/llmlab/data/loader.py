"""Memmap-backed next-token dataset loader with per-source mixing weights.

**Doc-boundary trade-off** (GPT-2/nanoGPT convention, applied here by default): a sampled
window is a random contiguous slice of a source's flat token array and MAY straddle a document
boundary — the `<|endoftext|>` token inside the window is the model's only signal that two
unrelated documents met, which is exactly how GPT-2 was trained ("concat-and-chunk"). The
alternative — clip every window to one document — throws away that boundary-modeling signal and
wastes the tail tokens of every document shorter than `seq_len`. Set
`Source.respect_doc_boundaries=True` per-source to opt into the stricter alternative for a
source whose documents are worth protecting (e.g. short, self-contained dictionary entries).

**Determinism** (needed for exact ablation comparability + resume): sampling is *stateless*
given `(seed, step)` — `get_batch(step, ...)` reseeds a fresh `numpy.random.Generator` from
`(seed, step)` rather than advancing a stored RNG. Resuming at step N reproduces exactly the
batches step N+1 onward would have produced uninterrupted, and there is nothing sampler-side to
checkpoint beyond the integer step counter. This mirrors nanoGPT's `torch.randint`-per-batch
approach (no epoch bookkeeping) — the right model for random-window sampling over a small,
many-times-repeated corpus rather than true single-pass epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Source:
    """One tokenized shard to draw training windows from."""

    name: str
    bin_path: Path
    weight: float
    respect_doc_boundaries: bool = False
    docstarts_path: Path | None = None  # required if respect_doc_boundaries=True

    def __post_init__(self) -> None:
        self.bin_path = Path(self.bin_path)
        if self.respect_doc_boundaries and self.docstarts_path is None:
            raise ValueError(f"source {self.name!r}: respect_doc_boundaries needs docstarts_path")
        if self.docstarts_path is not None:
            self.docstarts_path = Path(self.docstarts_path)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(
            name=d["name"],
            bin_path=d["path"],
            weight=d["weight"],
            respect_doc_boundaries=d.get("respect_doc_boundaries", False),
            docstarts_path=d.get("docstarts_path"),
        )


class MixedSourceLoader:
    """Combines one or more tokenized `.bin` sources into one (x, y) next-token stream.

    A single-source config (weight=1.0) is the common case (S-tier books+dictionary
    `train.bin`/`val.bin`) — mixing multiple sources by ratio is what RW-4 (domain data) and
    the M/L-tier TinyStories/FineWeb-Edu blend need, without a different code path.
    """

    def __init__(self, sources: list[Source], seq_len: int, seed: int):
        if not sources:
            raise ValueError("need at least one source")
        self.sources = sources
        self.seq_len = seq_len
        self.seed = seed

        self._arrays = [np.memmap(s.bin_path, dtype=np.uint16, mode="r") for s in sources]
        for arr, s in zip(self._arrays, sources):
            if len(arr) <= seq_len:
                raise ValueError(
                    f"source {s.name!r} has only {len(arr)} tokens, needs > seq_len={seq_len}"
                )

        weights = np.array([s.weight for s in sources], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("all mixing weights must be > 0")
        self._probs = weights / weights.sum()

        self._doc_spans: list[np.ndarray | None] = []  # (n_valid_docs, 2) [start, end) per source
        for arr, s in zip(self._arrays, sources):
            if not s.respect_doc_boundaries:
                self._doc_spans.append(None)
                continue
            starts = np.load(s.docstarts_path)
            ends = np.append(starts[1:], len(arr))
            spans = np.stack([starts, ends], axis=1)
            spans = spans[spans[:, 1] - spans[:, 0] > seq_len]  # need room for a full window
            if len(spans) == 0:
                raise ValueError(f"source {s.name!r}: no document is longer than seq_len={seq_len}")
            self._doc_spans.append(spans)

    def get_batch(
        self, step: int, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic given (seed, step): the same call always returns the same batch."""
        rng = np.random.default_rng([self.seed, step])
        src_idx = rng.choice(len(self.sources), size=batch_size, p=self._probs)

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, si in enumerate(src_idx):
            arr = self._arrays[si]
            spans = self._doc_spans[si]
            if spans is not None:
                span = spans[rng.integers(0, len(spans))]
                start = rng.integers(span[0], span[1] - self.seq_len)
            else:
                start = rng.integers(0, len(arr) - self.seq_len - 1)
            window = arr[start : start + self.seq_len + 1].astype(np.int64)
            x[i] = window[:-1]
            y[i] = window[1:]

        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        return x_t, y_t

    def fixed_eval_batches(
        self, n_batches: int, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-sample a fixed set of batches (steps 0..n_batches-1 of this loader's own,
        independent step namespace) for a stable eval loop — same batches every eval call."""
        return [self.get_batch(step=i, batch_size=batch_size, device=device) for i in range(n_batches)]
