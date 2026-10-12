"""Baseline 1: conventional rule-based triage on the build log only (no history, no commit, no LLM)."""

from __future__ import annotations

from .. import heuristics as H
from .base import Diagnosis, System


class RuleBased(System):
    name = "rule_based"

    # Priority order: infrastructure/config signatures are the most specific, generic exceptions the least.
    ORDER = ["environment", "dependency_config", "flaky", "test_bug"]

    def diagnose(self, rec) -> Diagnosis:
        for label in self.ORDER:
            ev = H.match_lines(rec.log, label)
            if ev:
                return Diagnosis(label, f"Log matches a known {label} signature.", ev, 1.0)
        kind = H.frame_kind(rec.log)
        if kind == "test":
            return Diagnosis("test_bug", "Failure is raised inside test code.", H.match_lines(rec.log, "product_bug"), 0.6)
        ev = H.match_lines(rec.log, "product_bug")
        return Diagnosis("product_bug", "Default: exception in production code.", ev, 0.5 if ev else 0.1)
