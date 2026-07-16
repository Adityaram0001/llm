"""Corpus-level perplexity and bits-per-byte on a tokenized `.bin` split.

**Why bits-per-byte, on top of perplexity:** perplexity is `exp(mean NLL per TOKEN)`, and a
token means something different for every tokenizer/vocab size — a 16k-vocab BPE token spans
more raw characters than a 50k-vocab one, so two models with different tokenizers can have wildly
different ppl on the *same* text despite modeling it equally well. Bits-per-byte instead measures
`(total NLL in bits) / (total BYTES of the original text)`, so it stays comparable across any
tokenizer choice — the metric this project will actually want once phase 9's L-tier run considers
a different vocab size (see the "v2 scale-up" parking-lot idea in PROGRESS.md).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tokenizers import Tokenizer

from llmlab.model import GPT


@torch.no_grad()
def evaluate_split(
    model: GPT,
    tokenizer: Tokenizer,
    bin_path: str,
    device: torch.device,
    seq_len: int | None = None,
    batch_size: int = 16,
) -> dict:
    """Non-overlapping-window perplexity + bits-per-byte over an entire `.bin` memmap file.

    Windows are `seq_len+1` tokens each (input `seq_len`, target `seq_len`), consumed with no
    stride/overlap and no cross-window context — the same "each window is independent" setting
    the model was trained under, so this measures exactly what training optimized. Any remainder
    tokens that don't fill a final window are dropped (a small fraction of a ~100K-token val
    split, immaterial to the result).
    """
    seq_len = seq_len or model.cfg.max_seq_len
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_windows = (len(tokens) - 1) // seq_len
    if n_windows == 0:
        raise ValueError(f"{bin_path} has only {len(tokens)} tokens, too short for seq_len={seq_len}")
    used = n_windows * seq_len + 1
    windows = np.lib.stride_tricks.as_strided(
        tokens, shape=(n_windows, seq_len + 1), strides=(tokens.strides[0] * seq_len, tokens.strides[0])
    ).astype(np.int64)

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, n_windows, batch_size):
        # `.copy()` materializes a normal contiguous array -- `windows` is a strided VIEW over
        # the memmap (consecutive windows deliberately overlap by one token, see above), and
        # feeding that non-contiguous layout straight into the model trips a `.view()` deep in
        # `GPT.forward` that assumes standard strides.
        batch = torch.from_numpy(windows[start : start + batch_size].copy()).to(device)
        # `.contiguous()`: slicing off one column from a shared (B, seq_len+1) tensor leaves
        # strides `GPT.forward`'s `.view(-1, ...)` can't flatten without a copy (unlike the
        # trainer's `inputs`/`targets`, which loader.py already materializes as two SEPARATE
        # (B, seq_len) arrays with no such offset).
        inputs, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        _, loss = model(inputs, targets)
        n = targets.numel()
        total_nll += float(loss) * n
        total_tokens += n

    text = tokenizer.decode(tokens[:used].tolist())
    n_bytes = len(text.encode("utf-8"))

    return {
        "ppl": math.exp(total_nll / total_tokens),
        "bits_per_byte": (total_nll / math.log(2)) / n_bytes,
        "n_tokens": total_tokens,
        "n_bytes": n_bytes,
    }
