"""Aggregate result rows into summary.json and a Markdown report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import metrics as M
from .taxonomy import LABELS

MOCK_BANNER = (
    "> **MOCK LLM / SYNTHETIC DATA WARNING.** If this run used `--llm mock` or the bundled synthetic sample, the numbers below only show "
    "that the pipeline works. They are not research results.\n"
)


def summarize(rows: list[dict], reference: str | None = None) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    systems = {}
    for name, rs in by_sys.items():
        rs = sorted(rs, key=lambda r: r["id"])
        gold, pred = [r["gold"] for r in rs], [r["pred"] for r in rs]
        s = M.summary(gold, pred)
        lo, hi = M.bootstrap_ci([r["correct"] for r in rs])
        ev_total = sum(r["grounded_evidence"] + r["ungrounded_evidence"] for r in rs)
        n = len(rs)
        s.update({
            "accuracy_ci95": [lo, hi],
            "confusion": M.confusion(gold, pred),
            "hallucination_rate": (sum(r["ungrounded_evidence"] for r in rs) / ev_total) if ev_total else None,
            "evidence_items": ev_total,
            "mean_latency_s": sum(r["latency_s"] for r in rs) / n,
            "total_cost_usd": sum(r["cost_usd"] for r in rs),
            "mean_cost_usd": sum(r["cost_usd"] for r in rs) / n,
            "mean_calls": sum(r["calls"] for r in rs) / n,
            "mean_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rs) / n,
            "errors": sum(1 for r in rs if r["pred"] == "error"),
            "invalid": sum(1 for r in rs if r["pred"] == "invalid"),
        })
        systems[name] = s
    ref = reference if reference in by_sys else (next(iter(by_sys)) if by_sys else None)
    tests = {}
    if ref:
        ref_correct = {r["id"]: r["correct"] for r in by_sys[ref]}
        for name, rs in by_sys.items():
            if name == ref:
                continue
            ids = [r["id"] for r in rs if r["id"] in ref_correct]
            a = {r["id"]: r["correct"] for r in rs}
            b, c, p = M.mcnemar_exact([a[i] for i in ids], [ref_correct[i] for i in ids])
            tests[name] = {"vs": ref, "only_this_right": b, "only_ref_right": c, "p_value": p}
    return {"reference": ref, "systems": systems, "mcnemar": tests}


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def write_report(summary: dict, out_dir, meta: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    sysd = summary["systems"]
    L = ["# Root-cause analysis: results", ""]
    if meta.get("mock") or meta.get("data_source") == "synthetic":
        L += [MOCK_BANNER]
    L += [f"- Records: **{meta.get('n_records')}**, folds: **{meta.get('folds')}** "
          f"({'grouped by project' if meta.get('group_by_project') else 'stratified'}), seed {meta.get('seed')}",
          f"- LLM: `{meta.get('llm')}`  |  data: `{meta.get('data')}`", "",
          "## Overall", "",
          "| System | Accuracy [95% CI] | Macro-F1 | Hallucination | Mean latency (s) | Mean cost ($) | Calls/record |",
          "|---|---|---|---|---|---|---|"]
    for name, s in sysd.items():
        lo, hi = s["accuracy_ci95"]
        L.append(f"| {name} | {_fmt(s['accuracy'], True)} [{_fmt(lo, True)}, {_fmt(hi, True)}] | {_fmt(s['macro_f1'])} | "
                 f"{_fmt(s['hallucination_rate'], True)} | {_fmt(s['mean_latency_s'], nd=3)} | {_fmt(s['mean_cost_usd'], nd=5)} | "
                 f"{_fmt(s['mean_calls'], nd=1)} |")
    L += ["", "Hallucination = share of cited evidence items that do not occur verbatim in the input "
          "(n/a for systems that cite no evidence). Cost is 0 unless `RCA_PRICE_IN_PER_MTOK` / `RCA_PRICE_OUT_PER_MTOK` are set.", ""]
    if summary["mcnemar"]:
        L += [f"## Paired significance vs `{summary['reference']}` (exact McNemar)", "",
              "| System | Only this right | Only reference right | p-value |", "|---|---|---|---|"]
        for name, t in summary["mcnemar"].items():
            L.append(f"| {name} | {t['only_this_right']} | {t['only_ref_right']} | {t['p_value']:.4f} |")
        L.append("")
    L += ["## Per-class F1", "", "| System | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for name, s in sysd.items():
        L.append(f"| {name} | " + " | ".join(_fmt(s["per_class"][lab]["f1"]) for lab in LABELS) + " |")
    L += ["", "Support: " + ", ".join(f"{lab}={next(iter(sysd.values()))['per_class'][lab]['support']}" for lab in LABELS) if sysd else "", ""]
    L += ["## Confusion matrices (rows = gold, columns = predicted)", ""]
    cols = LABELS + ["invalid"]
    for name, s in sysd.items():
        L += [f"### {name}", "", "| gold \\ pred | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for g in LABELS:
            L.append(f"| {g} | " + " | ".join(str(s["confusion"][g][c]) for c in cols) + " |")
        L.append("")
    path = out / "report.md"
    path.write_text("\n".join(L))
    return path
