"""Chat template: render/parse a conversation with the reserved special tokens, and encode it
into `(token_ids, supervise_mask)` for supervised fine-tuning (phase 8, Part A).

**Why a template at all?** A pretrained LM only continues text; it has no notion of "turns" or
"who is speaking." SFT teaches the model a *fixed protocol* — a few reserved marker tokens that
delimit user vs assistant spans — so that at inference we can hand it `<|user|>...<|assistant|>`
and it knows to produce an answer and then stop. This is exactly the chat-ML idea behind the
OpenAI/Anthropic messages APIs (`{"role": "user", ...}`), just spelled out at the token level.
The markers were reserved in the tokenizer back in phase 2 (D-014) precisely so their IDs never
shifted: `<|user|>`=2, `<|assistant|>`=3, `<|endoftext|>`=0, `<|pad|>`=1.

**The one template** (single- or multi-turn):

    <|user|>{q1}<|assistant|>{a1}<|endoftext|><|user|>{q2}<|assistant|>{a2}<|endoftext|>...

A user turn is `<|user|>{content}` (no terminator — the following `<|assistant|>` ends it); an
assistant turn is `<|assistant|>{content}<|endoftext|>`, where the trailing `<|endoftext|>` is
the learned *stop* signal. For inference, `add_generation_prompt=True` ends the string at a bare
`<|assistant|>` so the model continues from there (like chat-ML's generation prompt).

**Loss masking — THE mechanic of SFT** (see `sft_loader.py`): `encode_example` returns a
per-token `supervise` mask that is 1 only on assistant *content* tokens and the assistant-ending
`<|endoftext|>`. User turns and the `<|assistant|>` marker itself are context (0) — the model is
never trained to generate a user turn, only to answer one. Because each turn's content is
tokenized independently (`add_special_tokens=False`) and the marker IDs are spliced in by hand,
the mask is exact by construction — it never has to guess a boundary from separately-encoded
token counts (the fragile pattern behind eval bug RW-6).
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer

# Marker strings (canonical); IDs are always resolved from the passed tokenizer, never hardcoded.
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|endoftext|>"
PAD = "<|pad|>"


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {self.role!r}")


def to_messages(instruction: str, response: str | None = None) -> list[Message]:
    """Convenience for the common single-turn (instruction, [response]) SFT example."""
    msgs = [Message("user", instruction)]
    if response is not None:
        msgs.append(Message("assistant", response))
    return msgs


def render_chat(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """Human-readable string with literal markers (for notebooks / debugging / the REPL prompt).

    `add_generation_prompt=True` appends a bare `<|assistant|>` so a model can continue the
    assistant turn — use it when `messages` ends on a user turn and you want a completion.
    """
    parts: list[str] = []
    for m in messages:
        if m.role == "user":
            parts.append(f"{USER}{m.content}")
        else:
            parts.append(f"{ASSISTANT}{m.content}{EOT}")
    if add_generation_prompt:
        if messages and messages[-1].role != "user":
            raise ValueError("add_generation_prompt expects the conversation to end on a user turn")
        parts.append(ASSISTANT)
    return "".join(parts)


def encode_example(
    tokenizer: Tokenizer,
    messages: list[Message],
    *,
    add_generation_prompt: bool = False,
    supervise_eot: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode a conversation to `(ids, supervise)`, both length L and token-aligned.

    `supervise[i] == 1` iff `ids[i]` is a token the model should learn to produce: assistant
    *content* tokens, plus each assistant turn's terminating `<|endoftext|>` when `supervise_eot`
    (teaching the model to stop). Everything else — user markers/content and the `<|assistant|>`
    cue — is context, `supervise[i] == 0`.

    `add_generation_prompt=True` ends on a bare `<|assistant|>` (all-context) for inference; it is
    mutually exclusive with a trailing assistant turn.
    """
    uid = tokenizer.token_to_id(USER)
    aid = tokenizer.token_to_id(ASSISTANT)
    eid = tokenizer.token_to_id(EOT)
    for name, tid in [(USER, uid), (ASSISTANT, aid), (EOT, eid)]:
        if tid is None:
            raise ValueError(f"tokenizer is missing the reserved special token {name!r}")

    ids: list[int] = []
    sup: list[int] = []
    for m in messages:
        content = tokenizer.encode(m.content, add_special_tokens=False).ids
        if m.role == "user":
            ids.append(uid)
            sup.append(0)
            ids.extend(content)
            sup.extend([0] * len(content))
        else:  # assistant
            ids.append(aid)  # the cue is part of the prompt the model is answering, not a target
            sup.append(0)
            ids.extend(content)
            sup.extend([1] * len(content))
            ids.append(eid)
            sup.append(1 if supervise_eot else 0)

    if add_generation_prompt:
        if messages and messages[-1].role != "user":
            raise ValueError("add_generation_prompt expects the conversation to end on a user turn")
        ids.append(aid)
        sup.append(0)

    return ids, sup


def encode_prompt(tokenizer: Tokenizer, instruction: str) -> list[int]:
    """Token IDs for a single-turn inference prompt: `<|user|>{instruction}<|assistant|>`.

    Feed these to `GPT.generate`; decode the newly produced tokens up to `<|endoftext|>`.
    """
    ids, _ = encode_example(
        tokenizer, to_messages(instruction), add_generation_prompt=True
    )
    return ids


def describe_example(
    tokenizer: Tokenizer, ids: list[int], supervise: list[int]
) -> list[tuple[str, int]]:
    """`(token_string, supervised)` pairs for visualizing a masked example (notebook 09).

    Uses `id_to_token` (not `decode`) so byte-level artifacts like the leading-space marker 'Ġ'
    stay visible — the point is to *see* exactly which tokens carry loss."""
    return [(tokenizer.id_to_token(i), s) for i, s in zip(ids, supervise)]
