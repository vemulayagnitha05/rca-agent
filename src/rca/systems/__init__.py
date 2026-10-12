from __future__ import annotations

from .base import Diagnosis, System, Usage
from .llm_systems import MultiAgent, MultiAgentVerified, RagLLM, SingleLLM
from .rule import RuleBased
from .tfidf import TfidfClassifier

# Expected LLM calls per record, used for the cost guard in the CLI.
CALLS_PER_RECORD = {"rule": 0, "tfidf": 0, "single": 1, "rag": 1, "multi": 4, "multi_verifier": 5}

ALL = ["rule", "tfidf", "single", "rag", "multi", "multi_verifier"]


def factories(names, llm):
    """Return {name: zero-arg factory}. A fresh system is built per fold so no state leaks between folds."""
    table = {
        "rule": lambda: RuleBased(),
        "tfidf": lambda: TfidfClassifier(),
        "single": lambda: SingleLLM(llm),
        "rag": lambda: RagLLM(llm),
        "multi": lambda: MultiAgent(llm),
        "multi_verifier": lambda: MultiAgentVerified(llm),
    }
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown system(s): {unknown}; choose from {ALL}")
    return {n: table[n] for n in names}


__all__ = ["ALL", "CALLS_PER_RECORD", "Diagnosis", "System", "Usage", "factories"]
