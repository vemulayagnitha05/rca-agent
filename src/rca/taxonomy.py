"""Root-cause taxonomy. Change it here and every system, prompt and metric follows."""

from __future__ import annotations

LABELS = ["product_bug", "test_bug", "flaky", "environment", "dependency_config"]

DESCRIPTIONS = {
    "product_bug": "The production code is wrong; the test correctly exposes a defect.",
    "test_bug": "The test itself is wrong, outdated, badly mocked or badly set up; production code is fine.",
    "flaky": "Non-deterministic failure (timing, concurrency, ordering, randomness) with no relevant code change.",
    "environment": "Infrastructure problem: disk, memory, network, runner, container or external service.",
    "dependency_config": "Dependency, build or configuration problem: missing or incompatible packages, wrong versions, bad config.",
}

_ALIASES = {
    "bug": "product_bug",
    "code": "product_bug",
    "product": "product_bug",
    "source": "product_bug",
    "test": "test_bug",
    "flakiness": "flaky",
    "infra": "environment",
    "infrastructure": "environment",
    "env": "environment",
    "dependency": "dependency_config",
    "config": "dependency_config",
    "build": "dependency_config",
}


def normalize_label(value) -> str | None:
    """Map free text to a taxonomy label, or None if it is not a valid label."""
    if value is None:
        return None
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    s = _ALIASES.get(s, s)
    return s if s in LABELS else None
