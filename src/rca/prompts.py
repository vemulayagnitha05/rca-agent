"""Prompts and the case format shared by the real and the mock LLM."""

from __future__ import annotations

from .taxonomy import DESCRIPTIONS, LABELS
from .textutil import compact_log

KNOWN_SECTIONS = {
    "TEST", "LOG", "HISTORY", "COMMIT", "DIFF", "EXAMPLES", "PROPOSAL",
    "LOG ANALYST", "HISTORY ANALYST", "COMMIT ANALYST",
}

_TAXONOMY = "\n".join(f"- {lab}: {DESCRIPTIONS[lab]}" for lab in LABELS)
_LABEL_LIST = " | ".join(LABELS)

_EVIDENCE_RULE = (
    "Evidence must be short VERBATIM quotes copied exactly from the provided input "
    "(log lines, the history string, the commit message or diff headers). Never paraphrase or invent evidence."
)

SYSTEM = {
    "single": (
        "You are an expert in diagnosing software test failures. Decide the root cause of the failure below.\n"
        f"Root-cause categories:\n{_TAXONOMY}\n\n{_EVIDENCE_RULE}\n"
        f'Reply with JSON only: {{"label": "{_LABEL_LIST}", "hypothesis": "one or two sentences", '
        '"evidence": ["quote", ...], "confidence": 0.0-1.0}'
    ),
    "log_analyst": (
        "You are a log analyst. Read only the test name and build log and say which root-cause category the log "
        f"points to.\nCategories:\n{_TAXONOMY}\n\n{_EVIDENCE_RULE}\n"
        f'Reply with JSON only: {{"label_hint": "{_LABEL_LIST} | null", "confidence": 0.0-1.0, '
        '"findings": "short", "evidence": ["quote", ...]}'
    ),
    "history_analyst": (
        "You are a test-history analyst. The history lists prior outcomes of this test, oldest to newest "
        "(P = pass, F = fail), excluding the current failure. Intermittent mixes of P and F with no relevant change "
        f"suggest flakiness; a sudden failure after consistent passes suggests a regression.\nCategories:\n{_TAXONOMY}\n\n"
        f'{_EVIDENCE_RULE}\nReply with JSON only: {{"label_hint": "{_LABEL_LIST} | null", "confidence": 0.0-1.0, '
        '"findings": "short", "evidence": ["quote", ...]}'
    ),
    "commit_analyst": (
        "You are a change analyst. Look at the commit message and diff for the change that triggered this failure "
        f"and say which root-cause category it suggests.\nCategories:\n{_TAXONOMY}\n\n{_EVIDENCE_RULE}\n"
        f'Reply with JSON only: {{"label_hint": "{_LABEL_LIST} | null", "confidence": 0.0-1.0, '
        '"findings": "short", "evidence": ["quote", ...]}'
    ),
    "investigator": (
        "You are the lead investigator. Three analysts reported findings on a failing test. Weigh their reports "
        "against the raw case, resolve conflicts, and give the final root cause.\n"
        f"Categories:\n{_TAXONOMY}\n\n{_EVIDENCE_RULE}\n"
        f'Reply with JSON only: {{"label": "{_LABEL_LIST}", "hypothesis": "one or two sentences", '
        '"evidence": ["quote", ...], "confidence": 0.0-1.0}'
    ),
    "verifier": (
        "You are a skeptical reviewer of a root-cause diagnosis. Check whether the proposed label is supported by the "
        "case and whether a different category fits better. Be strict: accept only when the evidence supports the "
        f"label.\nCategories:\n{_TAXONOMY}\n\n"
        f'Reply with JSON only: {{"verdict": "accept | revise", "label": "{_LABEL_LIST}", "reason": "short"}}'
    ),
}
SYSTEM["rag"] = SYSTEM["single"] + (
    "\nSimilar past failures with their confirmed root causes are given under EXAMPLES. Use them as guidance only; "
    "base your answer on the current case."
)


def render_case(rec, *, sections=("TEST", "LOG", "HISTORY", "COMMIT", "DIFF"), max_log_chars: int = 6000) -> str:
    parts = []
    for sec in sections:
        if sec == "TEST":
            body = rec.test_name or "(unknown)"
        elif sec == "LOG":
            body = compact_log(rec.log, max_log_chars) or "(empty)"
        elif sec == "HISTORY":
            body = " ".join(rec.history) if rec.history else "(none)"
        elif sec == "COMMIT":
            body = rec.commit_message.strip() or "(none)"
        elif sec == "DIFF":
            body = rec.diff[:3000].strip() or "(none)"
        else:
            raise ValueError(sec)
        parts.append(f"### {sec}\n{body}")
    return "\n".join(parts)


def parse_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur, buf = None, []
    for line in text.splitlines():
        name = line[4:].strip().upper() if line.startswith("### ") else None
        if name in KNOWN_SECTIONS:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = name, []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out
