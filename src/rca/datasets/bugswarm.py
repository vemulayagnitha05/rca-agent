"""Converter for BugSwarm artifacts exported as JSON.

BugSwarm artifact fields used here were verified against the BugSwarm REST API
and a real artifact export.

IMPORTANT - labels are WEAK. BugSwarm's `classification` describes which
kinds of files the fix touched (source code / tests / build files), not why
the test failed. They are mapped here only as a starting point:
  code -> product_bug
  test only -> test_bug
  build only -> dependency_config

`flaky` and `environment` cannot be derived this way. Confirm or correct
labels manually before using them as ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import FailureRecord


def _yes(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "partial")


def weak_label(classification: dict) -> str | None:
    code, test, build = (
        _yes(classification.get(k))
        for k in ("code", "test", "build")
    )

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
        tag = str(
            art.get("image_tag")
            or art.get("_id")
            or f"artifact-{i}"
        )

        failed = art.get("failed_job") or {}
        passed = art.get("passed_job") or {}

        log = ""

        if logs_dir:
            for key in (tag, str(failed.get("job_id", ""))):
                p = Path(logs_dir) / f"{key}.log"

                if key and p.exists():
                    log = p.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    break

        cls = art.get("classification") or {}
        label = weak_label(cls)

        meta = dict(art.get("meta") or {})

        meta.update({
          "image_tag": tag,
          "classification": cls,
          "trigger_sha": failed.get("trigger_sha", ""),
          "failed_job_id": failed.get("job_id", ""),
          "passed_job_id": passed.get("job_id", ""),
          "failed_job": failed,
          "passed_job": passed,
          "metrics": art.get("metrics") or {},
          "test_framework": art.get("test_framework", ""),
          "pr_num": art.get("pr_num"),
          "ci_service": art.get("ci_service", ""),
          "build_system": art.get("build_system", ""),
          "base_branch": art.get("base_branch", ""),
          "branch": art.get("branch", ""),
          })

        out.append(
            FailureRecord(
                id=f"bugswarm-{tag}",
                source="bugswarm",
                project=str(art.get("repo", "")),
                language=str(art.get("lang", "")).lower(),
                test_name=str(failed.get("failed_tests") or ""),
                log=log,
                commit_message=str(failed.get("message", "")),
                diff="",
                history=[],
                label=label,
                label_source="bugswarm-weak" if label else "",
                meta=meta,
            )
        )

    return out
