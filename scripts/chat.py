#!/usr/bin/env python
"""Minimal chat REPL for a phase-8 SFT model — the payoff moment. 🎉

Loads a checkpoint, wraps each turn in the chat template (`<|user|>...<|assistant|>`), samples an
answer, and stops at the learned `<|endoftext|>`. Single-turn by default (each prompt is
independent, matching the dictionary-QA SFT data); pass --multi-turn to accumulate history.

Usage:
    python scripts/chat.py --run experiments/20260719_p8_sft-s-dictionary
    python scripts/chat.py --run <run> --temperature 0.7 --top-k 40
    echo "What does ephemeral mean?" | python scripts/chat.py --run <run>   # one-shot from a pipe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from llmlab.data.chat_format import EOT, Message, encode_example
from llmlab.model import GPT, ModelConfig
from llmlab.utils import get_device

ROOT = Path(__file__).resolve().parents[1]


def load(run_dir: Path, ckpt_name: str, device: torch.device) -> tuple[GPT, Tokenizer]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    tokenizer = Tokenizer.from_file(str(ROOT / cfg["tokenizer_dir"] / "tokenizer.json"))
    model_cfg = ModelConfig.from_yaml(str(ROOT / cfg["model_config"]))
    model = GPT(model_cfg).to(device)
    ckpt = torch.load(run_dir / "ckpt" / ckpt_name, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def respond(
    model: GPT, tokenizer: Tokenizer, history: list[Message], device: torch.device,
    temperature: float, top_k: int, max_new_tokens: int,
) -> str:
    ids, _ = encode_example(tokenizer, history, add_generation_prompt=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=True
    )
    gen = out[0].tolist()[len(ids):]
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id in gen:
        gen = gen[: gen.index(eot_id)]
    return tokenizer.decode(gen).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="experiments/<sft run folder>")
    parser.add_argument("--ckpt", type=str, default="best.pt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--multi-turn", action="store_true", help="accumulate conversation history")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load(args.run, args.ckpt, device)
    print(f"loaded {args.run.name} ({model.num_params() / 1e6:.1f}M params) on {device}. "
          "Ctrl-D or 'quit' to exit.\n")

    def answer(history: list[Message]) -> str:
        return respond(
            model, tokenizer, history, device,
            args.temperature, args.top_k, args.max_new_tokens,
        )

    # One-shot mode: stdin is a pipe, not a tty.
    if not sys.stdin.isatty():
        for line in sys.stdin:
            q = line.strip()
            if q:
                print(f"assistant: {answer([Message('user', q)])}\n")
        return

    history: list[Message] = []
    while True:
        try:
            q = input("you: ").strip()
        except EOFError:
            print()
            break
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        turn = history + [Message("user", q)] if args.multi_turn else [Message("user", q)]
        reply = answer(turn)
        print(f"assistant: {reply}\n")
        if args.multi_turn:
            history = turn + [Message("assistant", reply)]


if __name__ == "__main__":
    main()
