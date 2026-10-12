"""Manual labelling workflow: export a sheet, label it (ideally by two people), import, measure agreement."""

from __future__ import annotations

import csv
from pathlib import Path

from .metrics import cohen_kappa
from .taxonomy import LABELS, normalize_label

SHEET_COLUMNS = ["id", "source", "project", "test_name", "current_label", "label", "notes", "log_excerpt"]


def export_sheet(records, path, only_unlabeled: bool = False) -> int:
    n = 0
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in records:
            if only_unlabeled and r.label:
                continue
            w.writerow({"id": r.id, "source": r.source, "project": r.project, "test_name": r.test_name,
                        "current_label": r.label or "", "label": "", "notes": "",
                        "log_excerpt": r.log[:800].replace("\r", "")})
            n += 1
    return n


def read_sheet(path) -> dict[str, str]:
    """id -> normalised label for rows whose `label` column holds a valid taxonomy label."""
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = normalize_label(row.get("label"))
            if lab:
                out[row["id"]] = lab
    return out


def apply_sheet(records, path) -> tuple[int, list[str]]:
    """Write sheet labels into the records (label_source='manual'). Returns (applied, unknown ids)."""
    labels = read_sheet(path)
    by_id = {r.id: r for r in records}
    for rid, lab in labels.items():
        if rid in by_id:
            by_id[rid].label, by_id[rid].label_source = lab, "manual"
    return sum(1 for rid in labels if rid in by_id), [rid for rid in labels if rid not in by_id]


def kappa_from_sheets(path_a, path_b) -> dict:
    a, b = read_sheet(path_a), read_sheet(path_b)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two sheets share no labelled ids")
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    disagreements = [i for i in ids if a[i] != b[i]]
    return {"n": len(ids), "agreement": sum(x == y for x, y in zip(la, lb)) / len(ids),
            "kappa": cohen_kappa(la, lb), "disagreements": disagreements, "labels": LABELS}
