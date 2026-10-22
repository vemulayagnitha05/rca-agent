"""Command line interface:  python -m rca <command> --help"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import labeling
from .datasets import convert_bugswarm, convert_dir, convert_idoft, make_sample
from .evaluate import make_folds, run_experiment
from .llm import AnthropicLLM, CachedLLM, MockLLM
from .report import summarize, write_report
from .schema import load_jsonl, save_jsonl
from .systems import ALL, CALLS_PER_RECORD, factories


def _stats(records) -> str:
    lines = [f"records: {len(records)}"]
    for title, key in (("source", lambda r: r.source), ("label", lambda r: r.label or "(unlabeled)"),
                       ("label_source", lambda r: r.label_source or "-"), ("language", lambda r: r.language or "-")):
        lines.append(f"{title}: " + ", ".join(f"{k}={v}" for k, v in Counter(map(key, records)).most_common()))
    lines.append(f"projects: {len({r.project for r in records})}")
    lines.append(f"records with empty log: {sum(1 for r in records if not r.log.strip())}")
    return "\n".join(lines)


def cmd_make_sample(a):
    recs = make_sample(a.n, a.seed)
    save_jsonl(recs, a.out)
    print(f"wrote {len(recs)} SYNTHETIC records to {a.out} (pipeline testing only)")


def cmd_convert_idoft(a):
    recs = convert_idoft(a.csv, a.language)
    save_jsonl(recs, a.out)
    print(f"wrote {len(recs)} flaky-test records to {a.out} (no logs: IDoFT does not include them)")


def cmd_convert_bugswarm(a):
    recs = convert_bugswarm(a.json, a.logs_dir)
    save_jsonl(recs, a.out)
    print(f"wrote {len(recs)} records to {a.out}; labels are WEAK (fix-based), review them manually")
    print(_stats(recs))


def cmd_convert_dir(a):
    recs = convert_dir(a.dir, a.labels, a.source)
    save_jsonl(recs, a.out)
    print(f"wrote {len(recs)} records to {a.out}")
    print(_stats(recs))


def cmd_stats(a):
    recs = [r for p in a.data for r in load_jsonl(p)]
    print(_stats(recs))


def cmd_label_sheet(a):
    recs = [r for p in a.data for r in load_jsonl(p)]
    n = labeling.export_sheet(recs, a.out, a.only_unlabeled)
    print(f"wrote {n} rows to {a.out}. Fill the `label` column with one of: product_bug, test_bug, flaky, environment, dependency_config")


def cmd_apply_labels(a):
    recs = load_jsonl(a.data)
    applied, unknown = labeling.apply_sheet(recs, a.sheet)
    save_jsonl(recs, a.out)
    print(f"applied {applied} labels -> {a.out}" + (f"; {len(unknown)} ids not found" if unknown else ""))


def cmd_kappa(a):
    res = labeling.kappa_from_sheets(a.sheet_a, a.sheet_b)
    print(f"items: {res['n']}  agreement: {res['agreement']:.3f}  Cohen's kappa: {res['kappa']:.3f}")
    if res["disagreements"]:
        print("disagreements (resolve these, then update the labelling protocol):")
        for i in res["disagreements"]:
            print("  ", i)


def cmd_run(a):
    records = [r for p in a.data for r in load_jsonl(p)]
    n_all = len(records)
    labelled_records = [r for r in records if r.label]
    print(f"loaded {n_all} records, {len(labelled_records)} labelled")

    if not a.allow_empty_log:
        before = len(records)
        records = [r for r in records if r.log.strip()]
        if before != len(records):
            print(f"skipped {before - len(records)} records with an empty log (use --allow-empty-log to keep them)")
    if a.limit:
        records = records[: a.limit]
    if len(records) < a.folds:
        sys.exit(f"need at least {a.folds} usable records for {a.folds}-fold evaluation, got {len(records)}")

    names = [s.strip() for s in a.systems.split(",")]
    if a.llm == "mock":
        llm = MockLLM(fabricate=a.fabricate)
        print("NOTE: using the offline MockLLM (heuristic stand-in). Results are NOT research results.")
    else:
        llm = AnthropicLLM(model=a.model)
        est = sum(CALLS_PER_RECORD[n] for n in names) * len(records)
        print(f"estimated LLM calls: {est} (model {llm.model})")
        if est > a.max_calls:
            sys.exit(f"{est} calls exceeds --max-calls {a.max_calls}; raise it explicitly or lower --limit")
    out = Path(a.out)
    if a.llm != "mock" or a.cache:
        llm = CachedLLM(llm, a.cache or out / "llm_cache")

    def progress(fold, name, n):
        print(f"  fold {fold} {name}: {n} records", flush=True)

    rows = run_experiment(records, factories(names, llm), k=a.folds, seed=a.seed, workers=a.workers,
                          group_by_project=not a.no_group_by_project, progress=progress)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = summarize(rows, reference=a.reference)
    meta = {"n_records": len(records), "folds": a.folds, "seed": a.seed, "llm": getattr(llm, "name", "?"),
            "mock": a.llm == "mock", "data": ", ".join(a.data),
            "data_source": "synthetic" if all(r.source == "synthetic" for r in records) else "mixed",
            "group_by_project": not a.no_group_by_project, "systems": names}
    path = write_report(summary, out, meta)
    for name, s in summary["systems"].items():
        print(f"{name:22s} acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f}")
    print(f"report: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rca", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-sample", help="generate SYNTHETIC records for offline pipeline tests")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/sample.jsonl")
    s.set_defaults(fn=cmd_make_sample)

    s = sub.add_parser("convert-idoft", help="IDoFT pr-data.csv / gr-data.csv / py-data.csv -> JSONL")
    s.add_argument("--csv", required=True)
    s.add_argument("--language", default="java")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_convert_idoft)

    s = sub.add_parser("convert-bugswarm", help="BugSwarm JSON export (+ optional logs dir) -> JSONL with weak labels")
    s.add_argument("--json", required=True)
    s.add_argument("--logs-dir")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_convert_bugswarm)

    s = sub.add_parser("convert-dir", help="folder of .log files (GitBug-Java, LogChunks, your own) -> JSONL")
    s.add_argument("--dir", required=True)
    s.add_argument("--labels", help="CSV with columns id,label")
    s.add_argument("--source", default="dir")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_convert_dir)

    s = sub.add_parser("stats", help="dataset statistics")
    s.add_argument("data", nargs="+")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("label-sheet", help="export a CSV for manual labelling")
    s.add_argument("data", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--only-unlabeled", action="store_true")
    s.set_defaults(fn=cmd_label_sheet)

    s = sub.add_parser("apply-labels", help="write a filled labelling sheet back into a JSONL file")
    s.add_argument("--data", required=True)
    s.add_argument("--sheet", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_apply_labels)

    s = sub.add_parser("kappa", help="inter-annotator agreement between two filled sheets")
    s.add_argument("sheet_a")
    s.add_argument("sheet_b")
    s.set_defaults(fn=cmd_kappa)

    s = sub.add_parser("run", help="run the experiment")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--systems", default=",".join(ALL))
    s.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    s.add_argument("--model", help="override RCA_MODEL")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, help="use only the first N usable records")
    s.add_argument("--workers", type=int, default=1, help="parallel LLM requests")
    s.add_argument("--max-calls", type=int, default=3000, help="abort if the estimated number of LLM calls is higher")
    s.add_argument("--cache", help="LLM response cache dir (default: <out>/llm_cache for real LLMs)")
    s.add_argument("--fabricate", type=float, default=0.0, help="MockLLM only: probability of injecting fake evidence")
    s.add_argument("--reference", default="rule", help="system the McNemar tests compare against")
    s.add_argument("--no-group-by-project", action="store_true")
    s.add_argument("--allow-empty-log", action="store_true")
    s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
