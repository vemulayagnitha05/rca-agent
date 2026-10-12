"""LLM clients: a real Anthropic client, a disk cache for reproducibility, and an offline mock.

The MockLLM is a deterministic heuristic stand-in so the whole pipeline can run offline and in tests.
It is NOT a language model: numbers produced with it demonstrate the pipeline, not research findings.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import heuristics as H
from .prompts import parse_sections
from .taxonomy import LABELS


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in a model reply; None if there is none."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class AnthropicLLM:
    """Real client. Model and prices come from env vars so nothing is hard-coded:

    RCA_MODEL (default claude-sonnet-5-5), RCA_PRICE_IN_PER_MTOK, RCA_PRICE_OUT_PER_MTOK (USD; default 0
    which makes cost reports show 0 - set them from Anthropic's pricing page before reporting cost).
    """

    is_mock = False

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Install the SDK first:  pip install anthropic  (and set ANTHROPIC_API_KEY)") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model or os.environ.get("RCA_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.price_in = float(os.environ.get("RCA_PRICE_IN_PER_MTOK", "0"))
        self.price_out = float(os.environ.get("RCA_PRICE_OUT_PER_MTOK", "0"))
        self.name = f"anthropic:{self.model}"

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        try:
            resp = self.client.messages.create(temperature=0, **kwargs)
        except self._anthropic.BadRequestError:
            resp = self.client.messages.create(**kwargs)  # some models reject an explicit temperature
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        i, o = resp.usage.input_tokens, resp.usage.output_tokens
        cost = (i * self.price_in + o * self.price_out) / 1_000_000
        return LLMResponse(text, i, o, time.perf_counter() - t0, cost)


class CachedLLM:
    """Wrap any client with an on-disk cache so reruns are free and reproducible.

    A cache hit replays the original text, token counts, latency and cost, so reports stay comparable.
    """

    def __init__(self, inner, cache_dir):
        self.inner = inner
        self.is_mock = getattr(inner, "is_mock", False)
        self.name = getattr(inner, "name", "llm")
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        key = hashlib.sha256(json.dumps([self.name, role, system, user]).encode()).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return LLMResponse(**json.loads(path.read_text()))
        self.misses += 1
        resp = self.inner.complete(role, system, user)
        path.write_text(json.dumps(resp.__dict__))
        return resp


def _stable_rng(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16))


class MockLLM:
    """Deterministic offline stand-in. `fabricate` injects invented evidence to exercise the verifier."""

    is_mock = True
    name = "mock"

    def __init__(self, fabricate: float = 0.0):
        self.fabricate = fabricate

    # -- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _case(sec: dict[str, str]):
        hist = sec.get("HISTORY", "")
        history = [] if hist in ("", "(none)") else hist.split()
        none = lambda s: "" if s in ("", "(none)", "(empty)", "(unknown)") else s  # noqa: E731
        return none(sec.get("LOG", "")), history, none(sec.get("COMMIT", "")), none(sec.get("DIFF", ""))

    def _diagnose(self, sec, extra_votes: dict[str, float] | None = None):
        log, history, commit, diff = self._case(sec)
        scores = H.score_all(log, history, commit, diff)
        for lab, v in (extra_votes or {}).items():
            scores[lab] += v
        label, conf = H.argmax_label(scores)
        return {
            "label": label,
            "hypothesis": f"Signals most consistent with {label.replace('_', ' ')}.",
            "evidence": H.evidence_for(label, log, history, commit, diff),
            "confidence": conf,
        }

    # -- role handlers -----------------------------------------------------------------------
    def _single(self, sec, user):
        return self._diagnose(sec)

    def _rag(self, sec, user):
        votes: dict[str, float] = {}
        for lab in re.findall(r"^EXAMPLE_LABEL:\s*(\w+)", sec.get("EXAMPLES", ""), re.M):
            if lab in LABELS:
                votes[lab] = votes.get(lab, 0.0) + 0.5
        return self._diagnose(sec, votes)

    def _log_analyst(self, sec, user):
        log, _, _, _ = self._case(sec)
        scores = H.score_all(log, None, "", "")
        label, conf = H.argmax_label(scores)
        if conf == 0.0:
            return {"label_hint": None, "confidence": 0.0, "findings": "No decisive log signal.", "evidence": []}
        return {"label_hint": label, "confidence": conf, "findings": f"Log points to {label}.",
                "evidence": H.evidence_for(label, log, None, "", "")}

    def _history_analyst(self, sec, user):
        _, history, _, _ = self._case(sec)
        sig = H.history_signal(history)
        if sig == "flaky":
            return {"label_hint": "flaky", "confidence": 0.8, "findings": "Intermittent pass/fail history.",
                    "evidence": [" ".join(history)]}
        if sig == "regression":
            return {"label_hint": "product_bug", "confidence": 0.4, "findings": "Passed consistently before.",
                    "evidence": [" ".join(history)]}
        return {"label_hint": None, "confidence": 0.0, "findings": "History is uninformative.", "evidence": []}

    def _commit_analyst(self, sec, user):
        _, _, commit, diff = self._case(sec)
        hint, ev = H.commit_signal(commit, diff)
        if hint is None:
            return {"label_hint": None, "confidence": 0.0, "findings": "Change is uninformative.", "evidence": []}
        return {"label_hint": hint, "confidence": 0.6, "findings": f"Change suggests {hint}.", "evidence": ev}

    def _investigator(self, sec, user):
        totals = {lab: 0.0 for lab in LABELS}
        evidence: dict[str, list[str]] = {lab: [] for lab in LABELS}
        for name, weight in (("LOG ANALYST", 1.0), ("HISTORY ANALYST", 1.0), ("COMMIT ANALYST", 0.7)):
            try:
                rep = json.loads(sec.get(name, "{}"))
            except json.JSONDecodeError:
                continue
            hint = rep.get("label_hint")
            if hint in LABELS:
                totals[hint] += weight * float(rep.get("confidence", 0))
                evidence[hint].extend(rep.get("evidence", []))
        label, conf = H.argmax_label(totals)
        return {"label": label, "hypothesis": f"Analyst reports converge on {label.replace('_', ' ')}.",
                "evidence": evidence[label][:5], "confidence": conf}

    def _verifier(self, sec, user):
        log, history, commit, diff = self._case(sec)
        try:
            proposed = json.loads(sec.get("PROPOSAL", "{}")).get("label")
        except json.JSONDecodeError:
            proposed = None
        scores = H.score_all(log, history, commit, diff)
        best, _ = H.argmax_label(scores)
        if proposed in LABELS and best != proposed and scores[best] > scores[proposed] + 0.5:
            return {"verdict": "revise", "label": best, "reason": f"Signals favour {best} over {proposed}."}
        return {"verdict": "accept", "label": proposed or best, "reason": "Consistent with the case."}

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        sec = parse_sections(user)
        obj = getattr(self, f"_{role}")(sec, user)
        if self.fabricate and role in ("single", "rag", "investigator"):
            if _stable_rng(role, user).random() < self.fabricate:
                obj["evidence"] = list(obj.get("evidence", [])) + ["java.lang.FabricatedException: not in the input"]
        text = json.dumps(obj)
        return LLMResponse(text, (len(system) + len(user)) // 4, len(text) // 4, time.perf_counter() - t0, 0.0)
