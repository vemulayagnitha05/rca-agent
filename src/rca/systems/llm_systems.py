"""LLM-based systems: single prompt, RAG, multi-agent, and multi-agent + verifier."""

from __future__ import annotations

import json

from ..grounding import split_grounding
from ..llm import extract_json
from ..prompts import SYSTEM, render_case
from ..textutil import compact_log
from .base import Diagnosis, System, Usage, diagnosis_from_json
from .retrieval import Retriever


class LLMSystem(System):
    uses_llm = True

    def __init__(self, llm):
        self.llm = llm

    def _ask(self, usage: Usage, role: str, user: str) -> dict | None:
        resp = self.llm.complete(role, SYSTEM[role], user)
        usage.add(resp)
        return extract_json(resp.text)


class SingleLLM(LLMSystem):
    """One prompt that sees everything the multi-agent system sees (a fair monolithic baseline)."""

    name = "single_llm"

    def diagnose(self, rec) -> Diagnosis:
        usage = Usage()
        obj = self._ask(usage, "single", render_case(rec))
        return diagnosis_from_json(obj, usage)


class RagLLM(LLMSystem):
    """Single prompt plus the k most similar labelled failures from the training fold."""

    name = "rag_llm"

    def __init__(self, llm, k: int = 3):
        super().__init__(llm)
        self.k = k
        self.retriever = Retriever()

    def fit(self, train) -> None:
        self.retriever.fit(train)

    def diagnose(self, rec) -> Diagnosis:
        usage = Usage()
        examples = []
        for ex in self.retriever.query(rec, self.k):
            examples.append(f"EXAMPLE_LABEL: {ex.label}\nEXAMPLE_LOG: {compact_log(ex.log, 500)}")
        user = render_case(rec) + "\n### EXAMPLES\n" + ("\n---\n".join(examples) or "(none)")
        obj = self._ask(usage, "rag", user)
        return diagnosis_from_json(obj, usage)


class MultiAgent(LLMSystem):
    """Three specialist analysts (log, history, change) feed a lead investigator."""

    name = "multi_agent"

    def _investigate(self, rec, usage: Usage) -> Diagnosis:
        log_rep = self._ask(usage, "log_analyst", render_case(rec, sections=("TEST", "LOG")))
        hist_rep = self._ask(usage, "history_analyst", render_case(rec, sections=("TEST", "HISTORY")))
        commit_rep = self._ask(usage, "commit_analyst", render_case(rec, sections=("COMMIT", "DIFF")))
        user = (
            render_case(rec)
            + "\n### LOG ANALYST\n" + json.dumps(log_rep or {})
            + "\n### HISTORY ANALYST\n" + json.dumps(hist_rep or {})
            + "\n### COMMIT ANALYST\n" + json.dumps(commit_rep or {})
        )
        return diagnosis_from_json(self._ask(usage, "investigator", user), usage)

    def diagnose(self, rec) -> Diagnosis:
        usage = Usage()
        return self._investigate(rec, usage)


class MultiAgentVerified(MultiAgent):
    """Multi-agent + verifier.

    Step 1 (deterministic): drop evidence that does not occur in the input (grounding check).
    Step 2 (LLM critic): a skeptical reviewer may revise the label. Confidence is halved when no grounded
    evidence is left, so downstream consumers can see the diagnosis is weak.
    """

    name = "multi_agent_verifier"

    def diagnose(self, rec) -> Diagnosis:
        usage = Usage()
        diag = self._investigate(rec, usage)
        grounded, dropped = split_grounding(rec, diag.evidence)
        diag.evidence = grounded
        if dropped:
            diag.notes.append(f"verifier dropped {len(dropped)} ungrounded evidence item(s)")
        if not grounded:
            diag.confidence *= 0.5
            diag.notes.append("no grounded evidence remained")
        proposal = json.dumps({"label": diag.label, "hypothesis": diag.hypothesis, "evidence": grounded})
        verdict = self._ask(usage, "verifier", render_case(rec) + "\n### PROPOSAL\n" + proposal) or {}
        from ..taxonomy import normalize_label

        new_label = normalize_label(verdict.get("label"))
        if verdict.get("verdict") == "revise" and new_label and new_label != diag.label:
            diag.notes.append(f"verifier revised {diag.label} -> {new_label}: {verdict.get('reason', '')}")
            diag.label = new_label
        return diag
