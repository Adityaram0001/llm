"""Data factory — turn "I need N examples in format X" into a validated JSONL dataset.

Phase 7 of LLM-Lab. A backend-agnostic, config-driven pipeline: one task YAML per dataset,
one paranoid validator, and swappable generators (manual DeepSeek web / DeepSeek API /
local Gemma via Ollama or MLX). See `docs/phases/phase7_data_factory.md`.
"""
