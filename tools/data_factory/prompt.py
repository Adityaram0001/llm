"""Prompt construction — turn (task spec + a batch of seeds + a style axis) into a single
self-contained prompt string.

Every prompt is standalone (no reliance on chat history) so the manual DeepSeek workflow can
paste one batch per message and start a fresh chat any time without quality drift. The output
contract is strict: reply with ONLY a JSON array, one object per requested example, each
echoing its `seed_id` so replies map back to seeds for dedup + retry.
"""

from __future__ import annotations

import json

from .seeds import Seed
from .spec import TaskSpec


def _schema_description(schema: dict) -> str:
    """Render the schema block as compact, unambiguous field docs for the model."""
    lines = []
    for field, spec in schema.items():
        if isinstance(spec, dict):
            inner = ", ".join(f"{k}: {v}" for k, v in spec.items())
            lines.append(f'  "{field}": {{ {inner} }}')
        else:
            lines.append(f'  "{field}": {spec}')
    return "{\n" + ",\n".join(lines) + "\n}"


def build_prompt(task: TaskSpec, seeds: list[Seed], style: str) -> str:
    """Assemble the full prompt for one batch of seeds in one style.

    Layout is deliberately ordered invariant-first: the task instructions, schema, and few-shot
    example are byte-identical across every batch of a task, so keeping them as a contiguous
    prefix lets DeepSeek's automatic prefix caching hit on the whole block (cheaper input
    tokens). Only the per-batch varying parts — the style directive and the source items — come
    at the end, past the cache boundary.
    """
    schema_doc = _schema_description(task.schema)
    seed_lines = "\n".join(s.to_prompt_block() for s in seeds)

    few_shot_block = ""
    if task.few_shot:
        few_shot_block = (
            "\nEXAMPLE of the exact output format (one object per item):\n"
            + json.dumps(task.few_shot, ensure_ascii=False, indent=2)
            + "\n"
        )

    # --- Invariant prefix (identical across all batches of this task) ---
    prefix = f"""You are generating supervised fine-tuning data. Follow the instructions EXACTLY.

TASK:
{task.instructions}

OUTPUT SCHEMA — every object must match this shape:
{schema_doc}
Also include `"seed_id": "<the [id] of the source item>"` and set `meta.style` accordingly.
{few_shot_block}
STRICT OUTPUT RULES:
- Reply with ONLY a single JSON array. No prose before or after. No markdown code fences.
- Use straight double quotes, not smart quotes. Valid JSON (no trailing commas).
- Produce EXACTLY ONE object per source item, preserving each item's seed_id.
- Ground each answer ONLY in the material given — do not invent facts beyond it."""

    # --- Per-batch suffix (style + the source items) ---
    suffix = f"""

STYLE for THIS batch: {style}
(Write every instruction and response in a "{style}" register, and set meta.style to "{style}".)

SOURCE ITEMS ({len(seeds)} — produce exactly {len(seeds)} objects):
{seed_lines}
"""
    return prefix + suffix


def batch_seeds(seeds: list[Seed], per_prompt: int) -> list[list[Seed]]:
    """Split a flat seed list into fixed-size batches (last may be smaller)."""
    return [seeds[i:i + per_prompt] for i in range(0, len(seeds), per_prompt)]
