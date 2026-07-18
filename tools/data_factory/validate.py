"""Parse + validate + dedup — the paranoid gate between raw model text and `parsed/`.

DeepSeek and local models WILL occasionally wrap JSON in prose, add code fences, use smart
quotes, or leave trailing commas. We recover what we can, then validate hard against the task
schema and quality filters. Every rejected row is returned with a reason (never silently
dropped) so `factory.py` can route it to `failed/` and re-queue it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_SMART_QUOTES = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "″": '"', "′": "'",
}

_REFUSAL_MARKERS = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "i can not",
    "as an ai", "i'm unable", "i am unable", "i apologize", "cannot assist",
)


class ParseError(Exception):
    """The reply could not be coerced into a JSON array at all."""


@dataclass
class RowResult:
    ok: bool
    row: dict[str, Any] | None = None
    reason: str = ""


@dataclass
class IngestReport:
    valid: list[dict[str, Any]] = field(default_factory=list)
    invalid: list[RowResult] = field(default_factory=list)


# --------------------------------------------------------------------------- parsing

def extract_json_array(text: str) -> list[dict]:
    """Tolerantly pull a JSON array of objects out of raw model text.

    Handles: markdown code fences, leading/trailing prose, smart quotes, trailing commas.
    Raises ParseError if no array can be recovered.
    """
    s = text.strip()
    # Some local models (e.g. Gemma via Ollama) emit the SentencePiece meta-space U+2581 ('▁')
    # for indentation instead of real spaces — not valid JSON whitespace, so it breaks parsing.
    # It's never legitimate content in English text, so normalize it to a plain space.
    s = s.replace("▁", " ")
    # Strip ```json ... ``` fences if present.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    for bad, good in _SMART_QUOTES.items():
        s = s.replace(bad, good)

    # Isolate the outermost [ ... ] span.
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ParseError("no JSON array found in reply")
    candidate = s[start:end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Second chance: remove trailing commas before } or ].
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON: {e}") from e

    if not isinstance(data, list):
        raise ParseError("top-level JSON is not an array")
    rows = [r for r in data if isinstance(r, dict)]
    if not rows:
        raise ParseError("array contained no objects")
    return rows


# --------------------------------------------------------------------------- dot-paths

def resolve_path(row: dict, path: str) -> Any:
    """Resolve a dot-path like 'meta.word'. As a convenience, a single-segment path that's
    absent at top level is also looked up under 'meta' (so 'style' finds meta.style)."""
    parts = path.split(".")
    cur: Any = row
    for i, p in enumerate(parts):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif i == 0 and isinstance(row.get("meta"), dict) and p in row["meta"]:
            cur = row["meta"][p]
        else:
            return None
    return cur


def dedup_signature(row: dict, dedup_paths: list[str]) -> str:
    vals = [str(resolve_path(row, p)).strip().lower() for p in dedup_paths]
    return "||".join(vals)


# --------------------------------------------------------------------------- validation

def _lang_ok(text: str, want: str) -> bool:
    try:
        from langdetect import detect  # imported lazily; already a project dep
        return detect(text) == want
    except Exception:
        return True  # detection flaky on short text — don't reject on library failure


def validate_row(row: dict, task) -> RowResult:
    """Check one row against the task schema + quality filters. `task` is a TaskSpec."""
    # Required top-level fields present.
    for f in task.schema_fields:
        if f not in row:
            return RowResult(False, reason=f"missing field '{f}'")

    # Nested meta fields present + right shape.
    meta_spec = task.schema.get("meta")
    if isinstance(meta_spec, dict):
        meta = row.get("meta")
        if not isinstance(meta, dict):
            return RowResult(False, reason="meta is not an object")
        for mk in meta_spec:
            if mk not in meta or meta[mk] in (None, ""):
                return RowResult(False, reason=f"missing meta.{mk}")

    instruction = row.get("instruction")
    response = row.get("response")
    if not isinstance(instruction, str) or not isinstance(response, str):
        return RowResult(False, reason="instruction/response not strings")

    q = task.quality
    if not (q.min_instruction_chars <= len(instruction.strip()) <= q.max_instruction_chars):
        return RowResult(False, reason=f"instruction length {len(instruction.strip())} out of bounds")
    if not (q.min_response_chars <= len(response.strip()) <= q.max_response_chars):
        return RowResult(False, reason=f"response length {len(response.strip())} out of bounds")

    if q.forbid_refusals:
        low = response.lower()
        if any(m in low for m in _REFUSAL_MARKERS):
            return RowResult(False, reason="response looks like a refusal")

    if q.require_seed_term_in_response:
        term = resolve_path(row, f"meta.{task.seed_term_field}")
        if term and str(term).strip():
            # Check the RESPONSE specifically: the instruction usually echoes the term by
            # construction, so accepting it there would make this gate meaningless.
            if str(term).strip().lower() not in response.lower():
                return RowResult(False, reason=f"grounding term '{term}' absent from response")

    if q.require_lang and not _lang_ok(response, q.require_lang):
        return RowResult(False, reason=f"response not detected as '{q.require_lang}'")

    return RowResult(True, row=row)


def validate_rows(rows: list[dict], task, seen_signatures: set[str]) -> IngestReport:
    """Validate a batch of rows, deduping against `seen_signatures` (mutated in place)."""
    report = IngestReport()
    for row in rows:
        res = validate_row(row, task)
        if not res.ok:
            report.invalid.append(res)
            continue
        sig = dedup_signature(row, task.dedup_paths)
        if sig in seen_signatures:
            report.invalid.append(RowResult(False, row=row, reason=f"duplicate ({sig})"))
            continue
        seen_signatures.add(sig)
        report.valid.append(row)
    return report
