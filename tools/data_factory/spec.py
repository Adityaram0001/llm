"""Task specification — the contract for one dataset being built.

A task is a single YAML in `tasks/`. It fully describes what to generate (schema, count,
diversity axes), what to ground it in (seed source), and how to prompt for it (instructions
+ few-shot example). New dataset = new YAML, zero code (spec design rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QualityFilters:
    """Post-generation quality gates. Every field has a lenient default so a task YAML
    only overrides what it cares about."""

    min_response_chars: int = 10
    max_response_chars: int = 2000
    min_instruction_chars: int = 5
    max_instruction_chars: int = 500
    require_lang: str | None = "en"          # langdetect code, or None to skip
    forbid_refusals: bool = True             # drop "I'm sorry, I can't ..." style rows
    require_seed_term_in_response: bool = True  # response must mention the grounding term

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QualityFilters":
        return cls(**(d or {}))


@dataclass
class TaskSpec:
    """Parsed `tasks/<name>.yaml`. The single source of truth for one generation mission."""

    name: str
    target_count: int
    seed_source: str                 # path to a .jsonl (dictionary) or .txt/dir (book chunks)
    seed_kind: str                   # "dictionary" | "book_chunks"
    seeds_per_prompt: int
    schema: dict[str, Any]           # field -> type spec (str, or nested dict for meta)
    style_axes: list[str]
    dedup_key: str                   # e.g. "meta.word + style" — dot-paths joined by " + "
    instructions: str                # the human-language task description put in every prompt
    few_shot: list[dict[str, Any]]   # 1-3 example output rows shown to the model
    quality: QualityFilters = field(default_factory=QualityFilters)
    seed_term_field: str = "word"    # which seed field is "the term" (for grounding checks)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        path = Path(path)
        with path.open() as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            target_count=int(d["target_count"]),
            seed_source=d["seed_source"],
            seed_kind=d.get("seed_kind", "dictionary"),
            seeds_per_prompt=int(d.get("seeds_per_prompt", 10)),
            schema=d["schema"],
            style_axes=d.get("style_axes", ["formal"]),
            dedup_key=d["dedup_key"],
            instructions=d["instructions"].strip(),
            few_shot=d.get("few_shot", []),
            quality=QualityFilters.from_dict(d.get("quality")),
            seed_term_field=d.get("seed_term_field", "word"),
        )

    @property
    def dedup_paths(self) -> list[str]:
        """Dot-paths that jointly identify a row for dedup, e.g. ['meta.word', 'style']."""
        return [p.strip() for p in self.dedup_key.split("+")]

    @property
    def schema_fields(self) -> list[str]:
        """Top-level required output fields (e.g. instruction, response, meta)."""
        return list(self.schema.keys())


def find_task(name: str, tasks_dir: str | Path) -> TaskSpec:
    """Load task `name` from `<tasks_dir>/<name>.yaml`."""
    path = Path(tasks_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No task spec at {path}")
    return TaskSpec.from_yaml(path)
