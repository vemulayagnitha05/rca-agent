"""Small text helpers shared by systems."""

from __future__ import annotations

import re

_SALIENT = re.compile(r"error|exception|fail|assert|traceback|timed out|timeout|refused|denied|killed|no space", re.I)


def compact_log(log: str, max_chars: int = 6000) -> str:
    """Shorten a long build log while keeping the lines most likely to matter.

    Keeps the first lines, every salient line (errors, exceptions, assertions) and the tail,
    preserving original order. Short logs are returned unchanged.
    """
    if len(log) <= max_chars:
        return log
    lines = log.splitlines()
    keep = set(range(min(15, len(lines))))
    keep.update(range(max(0, len(lines) - 60), len(lines)))
    salient = [i for i, ln in enumerate(lines) if _SALIENT.search(ln)]
    for i in salient[:200]:
        keep.update(range(max(0, i - 1), min(len(lines), i + 3)))
    out, prev = [], -1
    for i in sorted(keep):
        if prev != -1 and i != prev + 1:
            out.append("[... omitted ...]")
        out.append(lines[i])
        prev = i
    text = "\n".join(out)
    if len(text) > max_chars:
        head = text[: max_chars // 3]
        tail = text[-(2 * max_chars) // 3 :]
        text = head + "\n[... omitted ...]\n" + tail
    return text


def case_text(rec, max_log_chars: int = 3000) -> str:
    """Flat text used by the TF-IDF classifier and the retriever."""
    return " ".join([rec.test_name, compact_log(rec.log, max_log_chars), rec.commit_message])
