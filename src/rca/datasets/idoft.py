"""Converter for the International Dataset of Flaky Tests (IDoFT).

Get the data:  https://github.com/TestingResearchIllinois/idoft  (files pr-data.csv, gr-data.csv, py-data.csv)

IDoFT lists flaky tests only (all positives for the `flaky` class) and contains no build logs. Use it to
(a) add flaky positives to a mixed dataset, or (b) study flaky sub-categories via meta["category"].
Logs must be collected by re-running the tests (e.g. with iDFlakies / NonDex) and attached afterwards.

Column names are matched by prefix so small header changes between releases do not break the converter;
check the first rows of your copy against the output of `rca stats`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import FailureRecord


def _col(header: list[str], *prefixes: str) -> str | None:
    for h in header:
        if any(h.strip().lower().startswith(p.lower()) for p in prefixes):
            return h
    return None


def convert_idoft(csv_path, language: str = "java") -> list[FailureRecord]:
    out: list[FailureRecord] = []
    with Path(csv_path).open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        c_url = _col(header, "project url")
        c_sha = _col(header, "sha")
        c_test = _col(header, "fully-qualified test name", "test name")
        c_cat = _col(header, "category")
        c_status = _col(header, "status")
        c_pr = _col(header, "pr link")
        if not (c_url and c_test):
            raise ValueError(f"{csv_path}: expected 'Project URL' and 'Fully-Qualified Test Name' columns, got {header}")
        for i, row in enumerate(reader):
            url = (row.get(c_url) or "").strip()
            test = (row.get(c_test) or "").strip()
            if not url or not test:
                continue
            out.append(FailureRecord(
                id=f"idoft-{i:05d}", source="idoft", project=url.rstrip("/").split("github.com/")[-1],
                language=language, test_name=test, label="flaky", label_source="idoft",
                meta={"category": (row.get(c_cat) or "").strip() if c_cat else "",
                      "status": (row.get(c_status) or "").strip() if c_status else "",
                      "sha": (row.get(c_sha) or "").strip() if c_sha else "",
                      "pr": (row.get(c_pr) or "").strip() if c_pr else ""},
            ))
    return out
