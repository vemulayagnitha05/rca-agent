"""Converter for BugSwarm artifacts exported as JSON.

Get the data:  https://www.bugswarm.org  (dataset browser, REST API and CLI; see their docs for export)
Source:        https://github.com/bugswarm/bugswarm

Expected input: a JSON list of artifact objects. Fields read (all optional, missing ones are tolerated):
  image_tag, repo, lang, failed_job.{job_id, trigger_sha}, classification.{build, code, test}
plus an optional directory of build logs named `<image_tag>.log` or `<failed_job.job_id>.log`.

IMPORTANT - labels are WEAK. BugSwarm's `classification` describes which kinds of files the *fix* touched
(source code / tests / build files), not why the test failed. They are mapped here only to give you a
starting point:   code -> product_bug,  test only -> test_bug,  build only -> dependency_config.
`flaky` and `environment` cannot be derived this way. Confirm or correct every label manually
(`rca label-sheet` / `rca apply-labels`) before using them as ground truth. The field names above come from
my reading of the BugSwarm docs and were NOT verified against a live export; adjust if yours differ.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import FailureRecord


def _yes(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "partial")


def weak_label(classification: dict) -> str | None:
    code, test, build = (_yes(classification.get(k)) for k in ("code", "test", "build"))
    if code:
        return "product_bug"
    if test:
        return "test_bug"
    if build:
        return "dependency_config"
    return None


def convert_bugswarm(json_path, logs_dir=None) -> list[FailureRecord]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("artifacts") or data.get("data") or [data]
    out: list[FailureRecord] = []
    for i, art in enumerate(data):
        tag = str(art.get("image_tag") or art.get("_id") or f"artifact-{i}")
        failed = art.get("failed_job") or {}
        log = ""
        if logs_dir:
            for key in (tag, str(failed.get("job_id", ""))):
                p = Path(logs_dir) / f"{key}.log"
                if key and p.exists():
                    log = p.read_text(encoding="utf-8", errors="replace")
                    break
        cls = art.get("classification") or {}
        label = weak_label(cls)
        out.append(FailureRecord(
            id=f"bugswarm-{tag}", source="bugswarm", project=str(art.get("repo", "")), language=str(art.get("lang", "")).lower(),
            log=log, label=label, label_source="bugswarm-weak" if label else "",
            meta={"image_tag": tag, "classification": cls, "trigger_sha": failed.get("trigger_sha", "")},
        ))
    return out
