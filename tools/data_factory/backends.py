"""Generation backends behind one interface.

A backend turns a prompt string into raw model text. Everything downstream (parse, validate,
dedup, export) is identical regardless of which backend produced the text — the whole point of
the phase-7 design.

  - ManualBackend    : the human is the transport (DeepSeek web chat). Not automated — the
                       `run` step is a person pasting into DeepSeek and saving the reply.
  - DeepSeekAPIBackend : DeepSeek's OpenAI-compatible HTTP API. Needs DEEPSEEK_API_KEY.
  - OllamaBackend    : local Gemma via the Ollama daemon (http://localhost:11434).
  - MLXBackend       : local Gemma via Apple MLX (mlx-lm), fastest on M-series silicon.

Local backends default to Gemma E4B per D-048; override the model tag per-run.
No CUDA/heavy deps: the two API-style backends use `requests` (already a project dep);
MLX is imported lazily so importing this module never requires mlx-lm to be installed.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod

import requests

# Default local model per D-048 (E4B = the grounded-generation workhorse on 16GB M4).
DEFAULT_OLLAMA_MODEL = "gemma3n:e4b"
DEFAULT_MLX_MODEL = "mlx-community/gemma-3n-E4B-it-4bit"
# deepseek-v4-flash (non-thinking mode) — the current name; the old `deepseek-chat` alias
# deprecates 2026-07-24 but auto-maps here. Non-thinking is correct for grounded generation
# (no wasted reasoning tokens). Overridable via DEEPSEEK_MODEL in .env or --model.
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class Backend(ABC):
    """A text generator. `generate(prompt)` returns the model's raw reply."""

    #: Automated backends are driven by `factory.py run`; manual is driven by a human.
    is_automated: bool = True
    name: str = "backend"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    def describe(self) -> str:
        return self.name


class ManualBackend(Backend):
    """DeepSeek web chat — the human pastes the prompt and saves the reply (D-004: no
    browser automation). There is no programmatic `generate`; batches flow through
    outbox/ -> (human) -> inbox/."""

    is_automated = False
    name = "manual"

    def generate(self, prompt: str) -> str:  # pragma: no cover - never called
        raise RuntimeError(
            "manual backend has no automated generate(): paste outbox/ prompts into DeepSeek "
            "web chat and save each reply into inbox/ with the matching batch number."
        )


class DeepSeekAPIBackend(Backend):
    """DeepSeek's OpenAI-compatible chat completions endpoint."""

    name = "api"

    # deepseek-v4-flash pricing (USD per 1M tokens), per the user 2026-07-18. Cache HITS are much
    # cheaper than misses — our invariant-prefix-first prompts maximize hits. (Hit price not
    # published to us; estimated at ~1/10 of miss, DeepSeek's usual ratio — output dominates cost
    # anyway so the estimate barely moves the total.)
    PRICE_IN_HIT = 0.014
    PRICE_IN_MISS = 0.14
    PRICE_OUT = 0.28

    def __init__(self, model: str | None = None, temperature: float = 1.0,
                 timeout: int = 180, max_tokens: int = 8192, thinking: bool = False):
        # Resolve model: explicit arg > DEEPSEEK_MODEL env (.env) > module default.
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        self.temperature = temperature
        self.timeout = timeout
        # High enough that a 60-item batch (~6k output tokens) never silently truncates.
        self.max_tokens = max_tokens
        # Grounded generation wants NON-thinking: bare `deepseek-v4-flash` defaults to THINKING
        # mode (burns output budget/cost on reasoning we don't need), so disable it explicitly.
        # Verified 2026-07-18: this yields reasoning_tokens=0 on deepseek-v4-flash.
        self.thinking = thinking
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        #: usage dict from the most recent call (cache-hit/miss + completion token counts).
        self.last_usage: dict | None = None
        #: running totals across this backend's lifetime, for budget awareness.
        self.totals = {"hit": 0, "miss": 0, "out": 0, "cost": 0.0}
        #: guards `totals` so concurrent `run --workers N` calls aggregate correctly.
        self._usage_lock = threading.Lock()
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Add it to .env (and get budget approval) before "
                "using --backend api, or use --backend manual / ollama / mlx instead."
            )

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "thinking": {"type": "enabled" if self.thinking else "disabled"},
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        self._record_usage(body.get("usage", {}))
        return body["choices"][0]["message"]["content"]

    def _record_usage(self, usage: dict) -> None:
        # DeepSeek reports prompt_cache_hit_tokens / prompt_cache_miss_tokens on top of the
        # OpenAI-standard fields; fall back gracefully if absent.
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens",
                         usage.get("prompt_tokens", 0) - hit)
        out = usage.get("completion_tokens", 0)
        cost = (hit * self.PRICE_IN_HIT + miss * self.PRICE_IN_MISS
                + out * self.PRICE_OUT) / 1_000_000
        usage = {"hit": hit, "miss": miss, "out": out, "cost": cost}
        with self._usage_lock:
            self.last_usage = usage
            for k in ("hit", "miss", "out", "cost"):
                self.totals[k] += usage[k]
        return usage

    def describe(self) -> str:
        return f"api:{self.model}"


class OllamaBackend(Backend):
    """Local Gemma via the Ollama daemon. Requires `ollama serve` running and the model
    pulled (`ollama pull gemma3n:e4b`)."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, temperature: float = 1.0,
                 timeout: int = 600):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": self.temperature}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def describe(self) -> str:
        return f"ollama:{self.model}"


class MLXBackend(Backend):
    """Local Gemma via Apple MLX (mlx-lm). Lazily imports mlx-lm on first use so the rest of
    the factory runs without it installed. Fastest local option on M-series."""

    name = "mlx"

    def __init__(self, model: str = DEFAULT_MLX_MODEL, temperature: float = 1.0,
                 max_tokens: int = 4096):
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from mlx_lm import load  # type: ignore
            except ImportError as e:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "mlx-lm not installed. `pip install mlx-lm` (Apple Silicon only) to use "
                    "--backend mlx, or use --backend ollama instead."
                ) from e
            self._model, self._tokenizer = load(self.model_id)

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        from mlx_lm import generate as mlx_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=self.temperature)
        return mlx_generate(self._model, self._tokenizer, prompt=formatted,
                            max_tokens=self.max_tokens, sampler=sampler, verbose=False)

    def describe(self) -> str:
        return f"mlx:{self.model_id}"


def get_backend(name: str, model: str | None = None, temperature: float = 1.0) -> Backend:
    """Construct a backend by name. `model` overrides the backend's default model tag."""
    name = name.lower()
    if name == "manual":
        return ManualBackend()
    if name == "api":
        # model=None lets the backend resolve DEEPSEEK_MODEL from .env, then the default.
        return DeepSeekAPIBackend(model=model, temperature=temperature)
    if name == "ollama":
        return OllamaBackend(model=model or DEFAULT_OLLAMA_MODEL, temperature=temperature)
    if name == "mlx":
        return MLXBackend(model=model or DEFAULT_MLX_MODEL, temperature=temperature)
    raise ValueError(f"Unknown backend {name!r} (expected manual|api|ollama|mlx)")
