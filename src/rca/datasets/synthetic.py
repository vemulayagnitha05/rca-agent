"""Synthetic failures for offline testing of the pipeline.

THIS IS NOT A RESEARCH DATASET. Logs are template-generated, so results on it say nothing about real CI
failures. It exists so the harness, metrics and CLI can be exercised without downloading anything.
About a quarter of the cases are deliberately ambiguous (generic log, signal only in history or commit).
"""

from __future__ import annotations

import random

from ..schema import FailureRecord
from ..taxonomy import LABELS

_PROJECTS = [("acme-billing", "java"), ("acme-search", "java"), ("shopkit", "python"), ("datapipe", "python"),
             ("geo-utils", "java"), ("authlib-lite", "python"), ("report-svc", "java"), ("mlflow-lite", "python")]
_CLASSES = ["Invoice", "Cart", "Parser", "Indexer", "Router", "Scheduler", "Cache", "Exporter"]
_METHODS = ["total", "load", "parse", "flush", "resolve", "apply", "render", "merge"]
_NOISE_JAVA = ["[INFO] Scanning for projects...", "[INFO] Building {p} 1.0-SNAPSHOT", "[INFO] --- maven-surefire-plugin:2.22.2:test ---",
               "[INFO] Running com.acme.{p}.{c}Test"]
_NOISE_PY = ["============================= test session starts ==============================", "platform linux -- Python 3.11.4",
             "collected 42 items", "tests/test_{c}.py F"]


def _pid(p: str) -> str:
    return p.replace("-", "")


def _frames_java(p, c, m, in_test):
    prod = f"\tat com.acme.{_pid(p)}.{c}.{m}({c}.java:{random.randint(20, 300)})"
    tst = f"\tat com.acme.{_pid(p)}.{c}Test.test{m.title()}({c}Test.java:{random.randint(10, 90)})"
    junit = "\tat org.junit.runners.ParentRunner.run(ParentRunner.java:413)"
    return "\n".join(([tst] if in_test else [prod, tst]) + [junit])


def _frames_py(p, c, m, in_test):
    prod = f'  File "src/{_pid(p)}/{c.lower()}.py", line {random.randint(10, 200)}, in {m}'
    tst = f'  File "tests/test_{c.lower()}.py", line {random.randint(5, 80)}, in test_{m}'
    return "Traceback (most recent call last):\n" + "\n".join(([tst] if in_test else [tst, prod]))


def _history(kind: str, rng: random.Random) -> list[str]:
    n = rng.randint(4, 8)
    if kind == "flaky":
        h = [rng.choice("PPF") for _ in range(n)]
        h[0], h[1] = "P", "F"
        h[2] = "P"
        return h
    h = ["P"] * n
    if rng.random() < 0.15:
        h[rng.randrange(n)] = "F"
    return h


def _make(i: int, label: str, rng: random.Random) -> FailureRecord:
    random.seed(rng.random())
    proj, lang = rng.choice(_PROJECTS)
    c, m = rng.choice(_CLASSES), rng.choice(_METHODS)
    ambiguous = rng.random() < 0.25
    noise = _NOISE_JAVA if lang == "java" else _NOISE_PY
    log = [ln.format(p=_pid(proj), c=c.lower() if lang == "python" else c) for ln in noise]
    history = _history(label, rng)
    commit, diff = "Refactor internals", f"diff --git a/src/{c}.src b/src/{c}.src\n+++ b/src/main/{c}.{'java' if lang == 'java' else 'py'}"
    ext = "java" if lang == "java" else "py"

    if label == "product_bug":
        if ambiguous:
            log.append("AssertionError: expected 120 but got 118" if lang == "python" else "java.lang.AssertionError: expected:<120> but was:<118>")
        elif lang == "java":
            log += [f"java.lang.NullPointerException: Cannot invoke \"String.length()\" because value is null", _frames_java(proj, c, m, False)]
        else:
            log += [_frames_py(proj, c, m, False), "KeyError: 'id'"]
    elif label == "test_bug":
        commit, diff = "Update tests for new API", f"diff --git a/src/test/{c}Test.{ext} b/src/test/{c}Test.{ext}\n+++ b/src/test/{c}Test.{ext}"
        if ambiguous:
            log.append("AssertionError: expected 5 but got 4" if lang == "python" else "java.lang.AssertionError: expected:<5> but was:<4>")
        elif lang == "java":
            log += ["org.mockito.exceptions.misusing.UnnecessaryStubbingException: Unnecessary stubbings detected", _frames_java(proj, c, m, True)]
        else:
            log += ["E       fixture 'db_session' not found", _frames_py(proj, c, m, True)]
    elif label == "flaky":
        commit, diff = "Update README", "diff --git a/README.md b/README.md\n+++ b/README.md"
        if ambiguous:
            log.append("AssertionError: expected True but got False" if lang == "python" else "java.lang.AssertionError: expected:<true> but was:<false>")
        elif lang == "java":
            log += [rng.choice(["org.awaitility.core.ConditionTimeoutException: timed out waiting for condition",
                                "java.util.ConcurrentModificationException"]), _frames_java(proj, c, m, True)]
        else:
            log += [rng.choice(["TimeoutError: timed out waiting for response", "RuntimeError: race condition detected"]), _frames_py(proj, c, m, True)]
    elif label == "environment":
        commit, diff = "Update CHANGELOG", "diff --git a/CHANGELOG.md b/CHANGELOG.md\n+++ b/CHANGELOG.md"
        if ambiguous:
            log.append("Process completed with exit code 1")
        else:
            log.append(rng.choice(["java.io.IOException: No space left on device", "ConnectionRefusedError: Connection refused",
                                   "java.net.UnknownHostException: repo.example.org", "OSError: [Errno 28] No space left on device"]))
    else:  # dependency_config
        commit, diff = f"Bump {c.lower()}-lib from 1.2 to 1.3", "diff --git a/requirements.txt b/requirements.txt\n+++ b/requirements.txt" if lang == "python" else "diff --git a/pom.xml b/pom.xml\n+++ b/pom.xml"
        if ambiguous:
            log.append("Build failed with exit code 1")
        elif lang == "java":
            log.append("[ERROR] Failed to execute goal on project: Could not resolve dependencies for project com.acme:app:jar:1.0")
        else:
            log.append(f"ModuleNotFoundError: No module named '{c.lower()}_lib'")

    return FailureRecord(
        id=f"syn-{i:04d}", source="synthetic", project=proj, language=lang,
        test_name=f"{c}Test.test{m.title()}" if lang == "java" else f"tests/test_{c.lower()}.py::test_{m}",
        log="\n".join(log), commit_message=commit, diff=diff, history=history, label=label, label_source="generator",
    )


def make_sample(n: int = 150, seed: int = 0) -> list[FailureRecord]:
    rng = random.Random(seed)
    return [_make(i, LABELS[i % len(LABELS)], rng) for i in range(n)]
