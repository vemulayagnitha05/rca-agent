"""Generic converter: a folder of log files (+ optional labels.csv) -> FailureRecords.

Use this for GitBug-Java (collect logs with `gitbug-java run`, see the test-results.json it writes) and for
LogChunks (logs are sorted by language and repository; labels must be added by you).

Layout:   <dir>/<project>/<anything>.log      (the immediate parent folder becomes the project)
Labels:   CSV with columns `id,label` where id is the file path relative to <dir> without extension.
Sidecars: <name>.commit.txt and <name>.diff next to a log are attached automatically if present.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import FailureRecord
from ..taxonomy import normalize_label


def convert_dir(root, labels_csv=None, source: str = "dir") -> list[FailureRecord]:
    root = Path(root)
    labels: dict[str, str] = {}
    if labels_csv:
        with Path(labels_csv).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lab = normalize_label(row.get("label"))
                if lab:
                    labels[row["id"].strip()] = lab
    out: list[FailureRecord] = []
    for p in sorted(list(root.rglob("*.log")) + list(root.rglob("*.txt"))):
        if p.name.endswith(".commit.txt"):
            continue
        rid = p.relative_to(root).with_suffix("").as_posix()
        commit, diff = p.with_name(p.stem + ".commit.txt"), p.with_name(p.stem + ".diff")
        label = labels.get(rid)
        out.append(FailureRecord(
            id=f"{source}-{rid}", source=source, project=p.parent.name if p.parent != root else "",
            log=p.read_text(encoding="utf-8", errors="replace"),
            commit_message=commit.read_text(encoding="utf-8", errors="replace") if commit.exists() else "",
            diff=diff.read_text(encoding="utf-8", errors="replace") if diff.exists() else "",
            label=label, label_source="manual" if label else "", meta={"path": str(p)},
        ))
    return out
