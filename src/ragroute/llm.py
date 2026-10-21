"""LLM clients and pricing.

`SimulatedLLM` stands in for a real model in offline benchmarks. Its price,
latency profile and per-difficulty accuracy are explicit parameters, so the
benchmark measures what caching and routing do *given* those assumptions.
`OpenAICompatibleLLM` calls any chat-completions endpoint (OpenAI, vLLM,
Together, Ollama, ...) for real runs.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_usd_per_mtok
                + output_tokens * self.output_usd_per_mtok) / 1e6


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    spec: ModelSpec

    def complete(self, system: str, user: str) -> Completion: ...


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


@dataclass(frozen=True)
class OracleAnswer:
    reference: str
    difficulty: str
    distractors: tuple[str, ...]


@dataclass
class LatencyProfile:
    ttft_s: float  # time to first token at ~1k prompt tokens
    s_per_output_token: float
    jitter_sigma: float = 0.25  # lognormal noise on the total


class SimulatedLLM:
    """Answers correctly with probability `accuracy[difficulty]`, else returns a
    plausible wrong answer. Token counts come from the real prompt; output length
    and latency are sampled."""

    def __init__(self, spec: ModelSpec, accuracy: dict[str, float], latency: LatencyProfile,
                 oracle: Callable[[str], OracleAnswer], output_tokens: dict[str, int],
                 seed: int = 0):
        self.spec = spec
        self.accuracy = accuracy
        self.latency = latency
        self.oracle = oracle
        self.output_tokens = output_tokens
        self._rng = random.Random(seed)

    def complete(self, system: str, user: str) -> Completion:
        truth = self.oracle(user)
        rng = self._rng
        correct = rng.random() < self.accuracy[truth.difficulty]
        text = truth.reference if correct else rng.choice(truth.distractors)
        in_tok = estimate_tokens(system) + estimate_tokens(user)
        out_tok = max(8, round(self.output_tokens[truth.difficulty] * rng.lognormvariate(0, 0.3)))
        lat = (self.latency.ttft_s * (0.7 + 0.3 * in_tok / 1000)
               + out_tok * self.latency.s_per_output_token)
        lat *= rng.lognormvariate(0, self.latency.jitter_sigma)
        return Completion(text, in_tok, out_tok, lat)


class OpenAICompatibleLLM:
    def __init__(self, spec: ModelSpec, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 300,
                 timeout_s: float = 60):
        self.spec = spec
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Completion:
        body = json.dumps({
            "model": self.spec.name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.load(resp)
        latency = time.perf_counter() - t0
        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"]
        return Completion(text,
                          usage.get("prompt_tokens") or estimate_tokens(system + user),
                          usage.get("completion_tokens") or estimate_tokens(text),
                          latency)
