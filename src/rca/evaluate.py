"""Experiment runner: leakage-aware cross-validation over all systems, per-record result rows."""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .grounding import split_grounding
from .systems.base import Diagnosis


def make_folds(records, k: int = 5, seed: int = 0, group_by_project: bool = True) -> list[list[int]]:
    """Return k lists of test indices."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[r.project or f"__solo_{i}"].append(i)

    if group_by_project and len(groups) >= k:
        items = list(groups.values())
        rng.shuffle(items)
        items.sort(key=len, reverse=True)

        for g in items:
            min(folds, key=len).extend(g)

        return folds

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_label[r.label or ""].append(i)

    j = 0
    for idxs in by_label.values():
        rng.shuffle(idxs)
        for i in idxs:
            folds[j % k].append(i)
            j += 1

    return folds


def _safe_diagnose(system, rec) -> Diagnosis:
    try:
        return system.diagnose(rec)
    except Exception as e:
        return Diagnosis("error", notes=[f"{type(e).__name__}: {e}"])


def run_experiment(
    records,
    factories: dict,
    k: int = 5,
    seed: int = 0,
    workers: int = 1,
    group_by_project: bool = True,
    progress=None,
) -> list[dict]:
    """Cross-validate every system. Returns one row per (system, record)."""
    folds = make_folds(records, k, seed, group_by_project)
    rows: list[dict] = []

    for fi, test_idx in enumerate(folds):
        if not test_idx:
            continue

        test_set = set(test_idx)
        train = [r for i, r in enumerate(records) if i not in test_set]
        test = [records[i] for i in test_idx]

        for name, make in factories.items():
            system = make()
            system.fit(train)

            if workers > 1 and system.uses_llm:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    diags = list(
                        pool.map(
                            lambda r: _safe_diagnose(system, r),
                            test,
                        )
                    )
            else:
                diags = [_safe_diagnose(system, r) for r in test]

            for rec, d in zip(test, diags):
                ok, bad = split_grounding(rec, d.evidence)
                rca_gt = rec.meta.get("rca_ground_truth") or {}
                gold_root_cause = rca_gt.get("root_cause_category")
                pred_root_cause = d.root_cause



                rows.append({
                    "id": rec.id,
                    "system": name,
                    "fold": fi,
                    "project": rec.project,
                    "gold": rec.label,
                    "gold_root_cause": gold_root_cause,
                    "pred_root_cause": pred_root_cause,
                    "root_cause_correct": int(
                       gold_root_cause is not None
                       and pred_root_cause == gold_root_cause
                    ),
                    "pred": d.label,
                    "correct": int(d.label == rec.label),
                    "confidence": d.confidence,
                    "hypothesis": d.hypothesis,
                    "evidence": d.evidence,
                    "grounded_evidence": len(ok),
                    "ungrounded_evidence": len(bad),
                    "latency_s": d.usage.latency_s,
                    "cost_usd": d.usage.cost_usd,
                    "input_tokens": d.usage.input_tokens,
                    "output_tokens": d.usage.output_tokens,
                    "calls": d.usage.calls,
                    "notes": d.notes,
                })

            if progress:
                progress(fi, name, len(test))

    return rows
