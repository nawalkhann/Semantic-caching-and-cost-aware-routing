"""Default model tiers and the simulated setup used by the benchmark and demo API.

Prices are per million tokens and modelled on typical small vs frontier model
tiers; latency and accuracy are assumptions, stated here so they can be changed.
"""

from __future__ import annotations

import re

from .corpus import Intent, build_intents
from .embeddings import tokenize
from .llm import LatencyProfile, ModelSpec, OracleAnswer, SimulatedLLM
from .service import question_from_prompt

SMALL = ModelSpec("small", input_usd_per_mtok=0.15, output_usd_per_mtok=0.60)
LARGE = ModelSpec("large", input_usd_per_mtok=3.00, output_usd_per_mtok=15.00)

SMALL_ACCURACY = {"easy": 0.96, "hard": 0.62}
LARGE_ACCURACY = {"easy": 0.98, "hard": 0.94}
SMALL_LATENCY = LatencyProfile(ttft_s=0.30, s_per_output_token=0.008)
LARGE_LATENCY = LatencyProfile(ttft_s=0.80, s_per_output_token=0.025)
# hard questions get worked answers (calculation shown), easy ones a sentence
OUTPUT_TOKENS = {"easy": 45, "hard": 170}


def _norm(q: str) -> str:
    return " ".join(tokenize(q))


class IntentOracle:
    """Maps a (possibly noisy) question back to its ground-truth intent.

    Lookup is by exact normalised text, which works because the workload is
    generated from known paraphrases; noise is added on top, so we index the
    noisy strings as the workload is produced (`register`)."""

    def __init__(self, intents: list[Intent] | None = None):
        self._by_query: dict[str, Intent] = {}
        for intent in intents or build_intents():
            for p in intent.paraphrases:
                self.register(p, intent)

    def register(self, query: str, intent: Intent) -> None:
        self._by_query[_norm(query)] = intent

    def intent(self, query: str) -> Intent | None:
        return self._by_query.get(_norm(query))

    def __call__(self, prompt: str) -> OracleAnswer:
        intent = self.intent(question_from_prompt(prompt))
        if intent is None:  # unknown question: treat as hard, no reference
            return OracleAnswer("I could not find that in the documentation.", "hard",
                                ("I could not find that in the documentation.",))
        return OracleAnswer(intent.answer, intent.difficulty, intent.distractors)


def simulated_models(oracle: IntentOracle, seed: int = 0) -> dict[str, SimulatedLLM]:
    return {
        SMALL.name: SimulatedLLM(SMALL, SMALL_ACCURACY, SMALL_LATENCY, oracle,
                                 OUTPUT_TOKENS, seed=seed),
        LARGE.name: SimulatedLLM(LARGE, LARGE_ACCURACY, LARGE_LATENCY, oracle,
                                 OUTPUT_TOKENS, seed=seed + 1),
    }


def grade(answer: str, intent: Intent) -> bool:
    """Correct if every required fact appears as a whole token ("30" does not match "300")."""
    text = answer.lower()
    return all(re.search(rf"(?<![\w.]){re.escape(f.lower())}(?!\w)", text)
               for f in intent.facts)
