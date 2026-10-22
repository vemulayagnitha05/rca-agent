"""Rule-based RCA using logs, history, commit/diff signals, and targeted heuristics."""

from __future__ import annotations

from .. import heuristics as H
from .base import Diagnosis, System


class RuleBased(System):
    name = "rule_based"

    # Priority order: infrastructure/config signatures are the most specific,
    # generic exceptions the least.
    ORDER = ["environment", "dependency_config", "flaky", "test_bug"]

    def diagnose(self, rec) -> Diagnosis:
        # First detect specific RCAs that have strong, direct evidence.
        root_cause, root_evidence = H.specific_root_cause(
            rec.commit_message,
            rec.diff,
            rec.log,
        )

        if root_cause == "environment_dependency_variance":
            return Diagnosis(
                label="test_bug",
                hypothesis=(
                    "Tests assert an environment-dependent MIME value "
                    "that varies with the libmagic version."
                ),
                evidence=root_evidence,
                confidence=1.0,
                root_cause=root_cause,
            )

        if root_cause == "python_version_compatibility":
            return Diagnosis(
                label="product_bug",
                hypothesis=(
                    "The failure is caused by code or tests requiring a newer "
                    "Python version than the CI environment provides."
                ),
                evidence=root_evidence,
                confidence=1.0,
                root_cause=root_cause,
            )

        if root_cause == "generated_artifact_stale":
            return Diagnosis(
                label="product_bug",
                hypothesis=(
                    "The checked-in generated artifact differs from the output "
                    "produced by its generator."
                ),
                evidence=root_evidence,
                confidence=1.0,
                root_cause=root_cause,
            )

        # Strong history-based flaky signal.
        if rec.history and H.history_signal(rec.history) == "flaky":
            ev = H.match_lines(rec.log, "flaky")
            if not ev:
                ev = ["history: " + " ".join(rec.history)]
            return Diagnosis(
                label="flaky",
                hypothesis="Prior outcomes show an intermittent failure pattern.",
                evidence=ev,
                confidence=0.9,
                root_cause=root_cause,
            )

        # Strong infrastructure signatures.
        ev = H.match_lines(rec.log, "environment")
        if ev:
            return Diagnosis(
                label="environment",
                hypothesis="The build log contains a known environment/infrastructure failure signature.",
                evidence=ev,
                confidence=1.0,
                root_cause=root_cause,
            )

        # Use the failed->passed diff before generic exception signatures.
        hint, hint_evidence = H.commit_signal(rec.commit_message, rec.diff)

        if hint == "dependency_config":
            return Diagnosis(
                label="dependency_config",
                hypothesis="The repair changes a dependency or build configuration file.",
                evidence=hint_evidence,
                confidence=0.9,
                root_cause=root_cause or "dependency_config",
            )

        if hint == "test_bug":
            return Diagnosis(
                label="test_bug",
                hypothesis="The repair is confined to test code, indicating a test-side defect or stale expectation.",
                evidence=hint_evidence,
                confidence=0.9,
                root_cause=root_cause or "test_bug",
            )

        if hint == "product_bug":
            return Diagnosis(
                label="product_bug",
                hypothesis="The repair changes production code, indicating a product-side defect.",
                evidence=hint_evidence,
                confidence=0.8,
                root_cause=root_cause or "product_bug",
            )

        # Only now use stack-frame classification.
        kind = H.frame_kind(rec.log)

        if kind == "test":
            return Diagnosis(
                label="test_bug",
                hypothesis="Failure is raised inside test code.",
                evidence=H.match_lines(rec.log, "product_bug"),
                confidence=0.6,
                root_cause=root_cause,
            )

        # Final log-based fallback.
        for label in ("dependency_config", "flaky", "test_bug", "product_bug"):
            ev = H.match_lines(rec.log, label)
            if ev:
                return Diagnosis(
                    label=label,
                    hypothesis=f"Log matches a known {label} signature.",
                    evidence=ev,
                    confidence=0.5,
                    root_cause=root_cause,
                )

        return Diagnosis(
            label="product_bug",
            hypothesis="Default: exception in production code.",
            evidence=[],
            confidence=0.1,
            root_cause=root_cause,
        )
