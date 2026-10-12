"""Evidence grounding: the basis of the hallucination metric.

A diagnosis cites evidence as verbatim snippets from its input. An evidence item is *grounded* if,
after whitespace and case normalisation, it occurs in the failure's input (test name, log, history,
commit message, diff). Ungrounded items are fabricated or misquoted. Hallucination rate for a
system = ungrounded evidence items / all evidence items it produced.
"""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _clean_item(s: str) -> str:
    return _norm(s.strip(" \t\"'`.…"))


def haystack(rec) -> str:
    return _norm(" ".join([rec.test_name, rec.log, " ".join(rec.history), rec.commit_message, rec.diff]))


def split_grounding(rec, evidence: list[str]) -> tuple[list[str], list[str]]:
    hay = haystack(rec)
    ok, bad = [], []
    for item in evidence:
        c = _clean_item(str(item))
        (ok if len(c) >= 3 and c in hay else bad).append(item)
    return ok, bad
