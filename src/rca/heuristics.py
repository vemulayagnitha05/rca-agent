"""Hand-written signals shared by the rule-based baseline and the offline mock LLM."""

from __future__ import annotations

import re

from .taxonomy import LABELS

_RAW_PATTERNS = {
    "environment": [
        r"no space left on device", r"connection (refused|reset by peer)", r"unknownhostexception",
        r"could not resolve host", r"java\.lang\.outofmemoryerror", r"too many open files", r"\benospc\b",
        r"503 service unavailable", r"cannot connect to the docker daemon", r"runner has received a shutdown signal",
        r"disk quota exceeded",
    ],
    "dependency_config": [
        r"could not resolve dependencies", r"modulenotfounderror", r"no matching distribution found",
        r"could not find artifact", r"failed to execute goal .*(dependency|enforcer|compiler)",
        r"importerror: cannot import name", r"pkg_resources\.distributionnotfound", r"dependency conflict",
        r"unsupported class file major version", r"missing required (config|environment variable)",
        r"nosuchmethoderror",
    ],
    "flaky": [
        r"concurrentmodificationexception", r"\bdeadlock\b", r"race condition", r"timeouterror",
        r"timed out waiting", r"timeoutexception", r"intermittent", r"\bflaky\b", r"order[- ]dependent",
        r"passed on retry", r"awaitility",
    ],
    "test_bug": [
        r"wanted but not invoked", r"unnecessarystubbingexception", r"fixture '.*' not found",
        r"outdated (test|snapshot)", r"snapshot .* mismatch", r"mockitoexception", r"hardcoded",
    ],
    "product_bug": [
        r"nullpointerexception", r"indexerror", r"zerodivisionerror", r"illegalargumentexception", r"keyerror",
        r"attributeerror", r"assertionerror", r"illegalstateexception", r"arrayindexoutofboundsexception",
        r"expected:<.*> but was:<.*>", r"typeerror",
    ],
}
LOG_PATTERNS = {k: [re.compile(p, re.I) for p in v] for k, v in _RAW_PATTERNS.items()}

LABEL_WEIGHT = {"environment": 2.0, "dependency_config": 2.0, "flaky": 1.5, "test_bug": 1.5, "product_bug": 0.8}

_JAVA_FRAME = re.compile(r"^\s*at\s+([\w$.]+)\.([\w$<>]+)\(([\w$]+\.java):(\d+)\)")
_PY_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)')
_JAVA_FRAMEWORK = ("org.junit.", "org.mockito.", "java.", "javax.", "jdk.", "sun.", "org.apache.maven.surefire", "org.testng.")
_PY_FRAMEWORK = ("site-packages", "_pytest", "pytest", "unittest", "/usr/lib/python")


def match_lines(text: str, label: str, limit: int = 3) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in LOG_PATTERNS[label]):
            s = line.strip()[:200]
            if s and s not in out:
                out.append(s)
            if len(out) >= limit:
                break
    return out


def frame_kind(log: str) -> str | None:
    """'test' or 'product' depending on where the failure was thrown, or None if no usable stack frame."""
    java, py = [], []
    for line in log.splitlines():
        m = _JAVA_FRAME.match(line)
        if m:
            ident = f"{m.group(1)} {m.group(3)}"
            if not ident.startswith(_JAVA_FRAMEWORK):
                java.append(ident)
            continue
        m = _PY_FRAME.match(line)
        if m and not any(f in m.group(1) for f in _PY_FRAMEWORK):
            py.append(m.group(1))
    ident = java[0] if java else (py[-1] if py else None)
    if ident is None:
        return None
    return "test" if re.search(r"test", ident, re.I) else "product"


def history_signal(history: list[str]) -> str | None:
    """'flaky' | 'regression' | 'persistent' | None, from prior outcomes (current failure excluded)."""
    h = [x.upper() for x in history if x]
    if len(h) < 3:
        return None
    transitions = sum(1 for a, b in zip(h, h[1:]) if a != b)
    if "P" in h and "F" in h and transitions >= 2:
        return "flaky"
    if all(x == "P" for x in h):
        return "regression"
    if all(x == "F" for x in h):
        return "persistent"
    return None


_DEP_FILES = re.compile(r"(pom\.xml|build\.gradle|requirements[\w.-]*\.txt|package(-lock)?\.json|dockerfile|setup\.py|pyproject\.toml)", re.I)
_DEP_MSG = re.compile(r"\b(bump|upgrade|update)\b.*\b(dependency|dependencies|version|requirements|pom|gradle|from \S+ to \S+)", re.I)


def changed_files(diff: str) -> list[str]:
    files = re.findall(r"^\+\+\+ b/(\S+)", diff, re.M)
    if not files:
        files = re.findall(r"^diff --git a/(\S+)", diff, re.M)
    return files


def commit_signal(commit_message: str, diff: str) -> tuple[str | None, list[str]]:
    """Weak label hint from the commit message / diff, plus the evidence line."""
    first = commit_message.strip().splitlines()[0] if commit_message.strip() else ""
    files = changed_files(diff)
    dep_file = next((f for f in files if _DEP_FILES.search(f)), None)
    if dep_file or (first and _DEP_MSG.search(first)):
        ev = [first] if first and _DEP_MSG.search(first) else [f"+++ b/{dep_file}"]
        return "dependency_config", ev
    if files:
        tests = [f for f in files if re.search(r"test", f, re.I)]
        if len(tests) == len(files):
            return "test_bug", [f"+++ b/{files[0]}"]
        src = [f for f in files if f not in tests and not re.search(r"\.(md|rst|txt)$", f, re.I)]
        if src:
            return "product_bug", [f"+++ b/{src[0]}"]
    return None, []


def score_all(log: str, history: list[str] | None, commit_message: str, diff: str) -> dict[str, float]:
    """Combine every signal into per-label scores. Pass history=None / empty strings to ignore a signal."""
    scores = {label: 0.0 for label in LABELS}
    for label in LABELS:
        scores[label] += LABEL_WEIGHT[label] * min(len(match_lines(log, label, limit=3)), 3) / 3 * 1.5
    fk = frame_kind(log)
    if fk == "test":
        scores["test_bug"] += 1.0
    elif fk == "product":
        scores["product_bug"] += 1.0
    if history:
        hs = history_signal(history)
        if hs == "flaky":
            scores["flaky"] += 2.0
        elif hs == "regression":
            scores["product_bug"] += 0.5
    hint, _ = commit_signal(commit_message, diff)
    if hint:
        scores[hint] += 1.0
    return scores


def argmax_label(scores: dict[str, float]) -> tuple[str, float]:
    """Best label and a 0..1 confidence; defaults to product_bug when there is no signal at all."""
    best = max(LABELS, key=lambda lab: (scores[lab], lab == "product_bug"))
    total = sum(scores.values())
    if scores[best] <= 0:
        return "product_bug", 0.0
    return best, round(scores[best] / total, 3)


def evidence_for(label: str, log: str, history: list[str] | None, commit_message: str, diff: str) -> list[str]:
    ev = match_lines(log, label, limit=3)
    if label == "flaky" and history and history_signal(history) == "flaky":
        ev.append(" ".join(history))
    hint, cev = commit_signal(commit_message, diff)
    if hint == label:
        ev.extend(cev)
    if not ev:
        for line in log.splitlines():
            if re.search(r"exception|error|fail", line, re.I):
                ev.append(line.strip()[:200])
                break
    return ev[:5]
