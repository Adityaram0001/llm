"""Tests for the phase-7 data factory (tools/data_factory/).

Focus on the paranoid validator — tolerant parsing of messy model replies, the quality gates,
and dedup — since that's where silent data corruption would otherwise creep in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The factory lives under tools/, not the installed llmlab package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from data_factory.seeds import Seed, _is_real_word, select_seeds  # noqa: E402
from data_factory.spec import TaskSpec  # noqa: E402
from data_factory.validate import (  # noqa: E402
    ParseError, dedup_signature, extract_json_array, resolve_path, validate_row, validate_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_TASK = PROJECT_ROOT / "tools" / "data_factory" / "tasks" / "sft_dictionary_qa.yaml"


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec.from_yaml(REAL_TASK)


def _row(word="ephemeral", style="formal", instruction="What does 'ephemeral' mean?",
         response="Something ephemeral lasts only a very short time before fading."):
    return {"seed_id": "dict-abc", "instruction": instruction, "response": response,
            "meta": {"word": word, "style": style}}


# --------------------------------------------------------------------------- parsing

def test_parse_clean_array():
    rows = extract_json_array('[{"a": 1}, {"b": 2}]')
    assert len(rows) == 2


def test_parse_strips_markdown_fences_and_prose():
    text = 'Sure! Here you go:\n```json\n[{"a": 1}]\n```'
    assert extract_json_array(text) == [{"a": 1}]


def test_parse_fixes_smart_quotes_and_trailing_comma():
    text = '[{“a”: “x”,},]'  # smart quotes + two trailing commas
    assert extract_json_array(text) == [{"a": "x"}]


def test_parse_normalizes_gemma_metaspace():
    # Gemma via Ollama indents with U+2581 ('▁'), which is not valid JSON whitespace.
    text = '[\n▁▁{\n▁▁▁▁"a": 1\n▁▁}\n]'
    assert extract_json_array(text) == [{"a": 1}]


def test_parse_no_array_raises():
    with pytest.raises(ParseError):
        extract_json_array("I'm sorry, I cannot produce JSON here.")


def test_parse_object_not_array_raises():
    with pytest.raises(ParseError):
        extract_json_array('{"a": 1}')


# --------------------------------------------------------------------------- validation

def test_valid_row_passes(task):
    assert validate_row(_row(), task).ok


def test_missing_field_rejected(task):
    row = _row()
    del row["response"]
    res = validate_row(row, task)
    assert not res.ok and "response" in res.reason


def test_missing_meta_key_rejected(task):
    row = _row()
    del row["meta"]["word"]
    res = validate_row(row, task)
    assert not res.ok and "meta.word" in res.reason


def test_refusal_rejected(task):
    res = validate_row(_row(response="I'm sorry, I can't help with that request."), task)
    assert not res.ok and "refusal" in res.reason


def test_grounding_checks_response_not_instruction(task):
    # Word is in the instruction (by construction) but NOT the response -> must reject,
    # otherwise the grounding gate is meaningless for this task.
    res = validate_row(_row(response="It denotes a concept that is hard to summarize."), task)
    assert not res.ok and "grounding" in res.reason


def test_too_short_response_rejected(task):
    res = validate_row(_row(response="ephemeral."), task)  # under min_response_chars
    assert not res.ok and "length" in res.reason


# --------------------------------------------------------------------------- dedup + paths

def test_dedup_rejects_same_word_same_style(task):
    seen: set[str] = set()
    r1 = _row(word="cat", style="formal", response="A cat is a small animal, the cat.")
    r2 = _row(word="cat", style="formal", response="A cat purrs; the cat is feline.")
    rep = validate_rows([r1, r2], task, seen)
    assert len(rep.valid) == 1
    assert any("duplicate" in inv.reason for inv in rep.invalid)


def test_dedup_allows_same_word_different_style(task):
    seen: set[str] = set()
    r1 = _row(word="cat", style="formal", response="The cat is a small feline animal.")
    r2 = _row(word="cat", style="casual", response="A cat's a fuzzy little critter, the cat.")
    rep = validate_rows([r1, r2], task, seen)
    assert len(rep.valid) == 2


def test_dedup_signature_and_resolve_path(task):
    row = _row(word="Cat", style="Formal")
    assert dedup_signature(row, task.dedup_paths) == "cat||formal"
    assert resolve_path(row, "meta.word") == "Cat"
    assert resolve_path(row, "style") == "Formal"  # single-segment falls back to meta


def test_select_seeds_diversifies_and_is_deterministic():
    # Simulate an alphabetically-sorted pool (all 'a' first, then 'b', ...).
    letters = "abcdefghij"
    pool = [Seed(id=f"{l}{i}", term=f"{l}{i}", text="x", raw={}) for l in letters for i in range(20)]
    # File order (no shuffle) starts all-'a' — the footgun.
    seq = select_seeds(pool, set(), 10, shuffle_seed=None)
    assert {s.term[0] for s in seq} == {"a"}
    # Shuffled spans many letters and is reproducible.
    a = select_seeds(pool, set(), 10, shuffle_seed=1337)
    b = select_seeds(pool, set(), 10, shuffle_seed=1337)
    assert [s.id for s in a] == [s.id for s in b]           # deterministic
    assert len({s.term[0] for s in a}) >= 5                 # diverse first-letters


def test_select_seeds_excludes_done():
    pool = [Seed(id=f"s{i}", term=f"w{i}", text="x", raw={}) for i in range(10)]
    got = select_seeds(pool, exclude_ids={"s0", "s1", "s2"}, count=100, shuffle_seed=1337)
    assert len(got) == 7 and not ({"s0", "s1", "s2"} & {s.id for s in got})


def test_seed_word_filter():
    # Real vocabulary words pass; the numerals/abbreviations heading the sorted GCIDE dump don't.
    assert _is_real_word("aardvark")
    assert _is_real_word("first-class")  # hyphen allowed
    assert not _is_real_word("1")
    assert not _is_real_word("10")
    assert not _is_real_word("1-dodecanol")
    assert not _is_real_word("1st-class")


def test_task_spec_parses_real_yaml(task):
    assert task.name == "sft_dictionary_qa"
    assert task.dedup_paths == ["meta.word", "meta.style"]
    assert "formal" in task.style_axes
