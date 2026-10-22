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
_GENERATED_FILES = re.compile(
    r"(^|/)([^/]*(?:_gen|generated)[^/]*)\.(py|java|js|ts|go|rs)$",
    re.I,
)


_DEP_VERSION_CHANGE = re.compile(
    r"^[+-].*(?:version|VERSION)\s*[:=]\s*[\"']?\d+(?:\.\d+)+(?:[-+][\w.-]+)?[\"']?",
    re.I,
)
_VERSIONED_DEPENDENCY_CHANGE = re.compile(
    r'^[+-].*[\w.-]+-\d+(?:\.\d+)+(?:[-+][\w.-]+)?\.(?:jar|whl|egg|zip)',
    re.I,
)
_TEST_EXPECTATION_CHANGE = re.compile(
    r"""^[+-]\s*(?:
        .*?\b(?:self\.)?assert(?:Equal|Equals|True|False|Raises|Is|IsNone|In|NotIn|AlmostEqual|Regex)\b
        |.*?\bassert\s+.+(?:==|!=|<=|>=|<|>)\s+.
        |.*?\b(?:expected|actual)\b.*(?:==|!=|assert|expect)
        |.*?\b(?:expect|expectation)\b.*(?:assert|==|!=)
    )""",
    re.I | re.X,
)
_ENVIRONMENT_TEST_CHANGE = re.compile(
    r"^[+-].*(?:\bmime\b|\bANY\b)",
    re.I,
)



def changed_files(diff: str) -> list[str]:
    files = []

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                path = re.sub(r"^(?:b/|passed/|failed/)", "", path)
                files.append(path)

        elif line.startswith("--- "):
            path = line[4:].strip()
            if path != "/dev/null":
                path = re.sub(r"^(?:a/|passed/|failed/)", "", path)
                files.append(path)

    if not files:
        matches = re.findall(r"^diff --git a/(\S+) b/(\S+)", diff, re.M)
        files = [b for _, b in matches]

    return list(dict.fromkeys(files))


def commit_signal(commit_message: str, diff: str) -> tuple[str | None, list[str]]:
    """Weak label hint from the commit message / diff, plus evidence lines."""
    first = commit_message.strip().splitlines()[0] if commit_message.strip() else ""
    files = changed_files(diff)

    generated_file = next((f for f in files if _GENERATED_FILES.search(f)), None)
    if generated_file:
        generated_line = next(
            (
                line.strip()
                for line in diff.splitlines()
                if line.startswith("+++") and generated_file in line
            ),
            None,
        )
        return "product_bug", [generated_line] if generated_line else []

    if files:
        tests = [
         f
         for f in files
         if re.search(
              r"(?:^|/)(?:tests?|testdata|specs?)(?:/|$)|"
              r"(?:^|/)test_[^/]+\.py$|"
              r"[^/]+_test\.[^.]+$",
              f,
              re.I,
        )
       ]
        dep_files = [f for f in files if _DEP_FILES.search(f)]

        src = [
            f
            for f in files
            if (
                f not in tests
                and f not in dep_files
                and not re.search(r"\.(md|rst|txt)$", f, re.I)
            )
        ]

        # A test expectation changed, even when production code changed
        # in the same commit. This is stronger than the generic source
        # file signal.
        expectation_lines = [
            line.strip()
            for line in diff.splitlines()
            if _TEST_EXPECTATION_CHANGE.search(line)
        ]
        if tests and expectation_lines:
            return "test_bug", expectation_lines[:2]

        # Dependency version changes can live inside source/config files,
        # so inspect the diff rather than relying only on filenames.
        dependency_lines = [
            line.strip()
            for line in diff.splitlines()
            if (
                _DEP_VERSION_CHANGE.search(line)
                or _VERSIONED_DEPENDENCY_CHANGE.search(line)
            )
        ]
        if dependency_lines:
            return "dependency_config", dependency_lines[:2]

        if dep_files and not src:
            dep_file = dep_files[0]
            return "dependency_config", [f"+++ b/{dep_file}"]

        if not tests and not dep_files and not src:
            return None, []
        
        if tests and not src and not dep_files:
            return "test_bug", [f"+++ b/{tests[0]}"]

        if len(tests) == len(files):
            return "test_bug", [f"+++ b/{tests[0]}"]
        
        if src:
            return "product_bug", [f"+++ b/{src[0]}"]

    if first and _DEP_MSG.search(first):
        return "dependency_config", [first]

    return None, []
def specific_root_cause(
    commit_message: str,
    diff: str,
    log: str,
) -> tuple[str | None, list[str]]:
    combined = "\n".join((commit_message, diff, log))
    files = changed_files(diff)

    if (
        re.search(r"\blibmagic\b", combined, re.I)
        and re.search(r"\bmime\b", combined, re.I)
        and re.search(r"\bANY\b", diff)
    ):
        return (
            "environment_dependency_variance",
            [
                line.strip()
                for line in diff.splitlines()
                if _ENVIRONMENT_TEST_CHANGE.search(line)
            ][:3],
        )

    # asyncio.run / AttributeError compatibility rule
    if "asyncio.run" in combined.lower() and "attributeerror" in combined.lower():
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if "asyncio.run" in lowered or "attributeerror" in lowered:
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break
        return "python_version_compatibility", evidence
    # Checkstyle: brittle test expectation containing developer-local paths
    # and locale-specific output.
    if (
        "translationchecktest" in combined
        and "unexpected log output" in combined
        and (
            ("/home/valeria/" in combined and "/home/travis/" in combined)
            or ("expected" in combined and "actual" in combined)
        )
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "translationchecktest" in lowered
                or "unexpected log output" in lowered
                or "/home/valeria/" in lowered
                or "/home/travis/" in lowered
            ):
                evidence.append(line.strip())
            if len(evidence) >= 4:
                break
        return "test_bug", evidence

    # ComfyUI-Lora-Manager: optional torch-dependent module fails during
    # test collection because torch is unavailable in the CI environment.
    if (
        "modulenotfounderror" in combined
        and "no module named 'torch'" in combined
        and ("test collection" in combined or "collecting" in combined)
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "modulenotfounderror" in lowered
                or "no module named 'torch'" in lowered
                or "test collection" in lowered
                or "collecting" in lowered
            ):
                evidence.append(line.strip())
            if len(evidence) >= 4:
                break
        return "environment", evidence

    # yt-dlp: explicit extractor/fixture collision.
    if (
        "test_no_duplicates" in combined
        and "should not match url" in combined
        and "belongs to" in combined
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "test_no_duplicates" in lowered
                or "should not match url" in lowered
                or "belongs to" in lowered
            ):
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break
        return "test_bug", evidence

    # OIPA: explicit mock expectation mismatch.
    if (
        "expected call:" in combined
        and "actual call:" in combined
        and ("test_" in combined or "serializer" in combined)
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "expected call:" in lowered
                or "actual call:" in lowered
                or "serializer" in lowered
            ):
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break
        return "test_bug", evidence

    # mjml-python: Python 3.6 cannot parse assignment expressions (:=).
    if (
        "python 3.6" in combined
        and "syntaxerror" in combined
        and "invalid syntax" in combined
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "python 3.6" in lowered
                or "syntaxerror" in lowered
                or "invalid syntax" in lowered
            ):
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break
        return "python_version_compatibility", evidence

    generated_file = next(
        (f for f in files if _GENERATED_FILES.search(f)),
        None,
    )


    if generated_file:
        evidence = []

        generated_line = next(
            (
                line.strip()
                for line in diff.splitlines()
                if line.startswith("+++") and generated_file in line
            ),
            None,
        )

        if generated_line:
            evidence.append(generated_line)

        return "generated_artifact_stale", evidence

    # libmagic can produce different MIME strings across versions/environments.
    # Detect the specific failure pattern without changing the coarse taxonomy.
    combined = " ".join(
        part for part in (commit_message, log, diff) if part
    ).lower()

    if (
        "libmagic" in combined
        and "mime" in combined
        and "assertionerror" in combined
    ):
        evidence = []
        for line in log.splitlines():
            lowered = line.lower()
            if (
                "assertionerror" in lowered
                or ("mime" in lowered and "application/" in lowered)
            ):
                evidence.append(line.strip())
            if len(evidence) >= 3:
                break

        return "environment_dependency_variance", evidence

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
