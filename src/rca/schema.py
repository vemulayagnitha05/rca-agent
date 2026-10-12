"""Canonical failure record. Every dataset converter produces this format (JSONL)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FailureRecord:
    id: str
    source: str = ""  # e.g. bugswarm, gitbug-java, idoft, logchunks, synthetic
    project: str = ""  # used to build leakage-free folds
    language: str = ""
    test_name: str = ""
    log: str = ""
    commit_message: str = ""
    diff: str = ""
    history: list[str] = field(default_factory=list)  # prior outcomes oldest->newest, "P"/"F"
    label: str | None = None  # gold root cause (see taxonomy)
    label_source: str = ""  # manual | bugswarm-weak | idoft | generator ...
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def save_jsonl(records, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path) -> list[FailureRecord]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(FailureRecord.from_dict(json.loads(line)))
    return out
