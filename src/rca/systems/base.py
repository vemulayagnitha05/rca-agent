from __future__ import annotations

from dataclasses import dataclass, field

from ..taxonomy import normalize_label


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, resp) -> None:
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cost_usd += resp.cost_usd
        self.latency_s += resp.latency_s
        self.calls += 1


@dataclass
class Diagnosis:
    label: str  # a taxonomy label, or "invalid" / "error"
    hypothesis: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    usage: Usage = field(default_factory=Usage)
    notes: list[str] = field(default_factory=list)
    root_cause: str | None = None


class System:
    """A root-cause analysis approach. `fit` is a no-op for systems that need no training data."""

    name = "base"
    uses_llm = False

    def fit(self, train) -> None:
        pass

    def diagnose(self, rec) -> Diagnosis:
        raise NotImplementedError


def diagnosis_from_json(obj: dict | None, usage: Usage) -> Diagnosis:
    obj = obj or {}
    label = normalize_label(obj.get("label")) or "invalid"
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    return Diagnosis(
        label,
        str(obj.get("hypothesis", "")),
        [str(e) for e in evidence][:5],
        conf,
        usage,
        root_cause=obj.get("root_cause"),
    )
