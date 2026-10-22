import json
import tempfile
import unittest
from pathlib import Path

from rca import heuristics as H
from rca import labeling
from rca import metrics as M
from rca.datasets import convert_bugswarm, convert_dir, convert_idoft, make_sample
from rca.datasets.bugswarm import weak_label
from rca.evaluate import make_folds, run_experiment
from rca.grounding import split_grounding
from rca.llm import CachedLLM, MockLLM, extract_json
from rca.prompts import parse_sections, render_case
from rca.report import summarize, write_report
from rca.schema import FailureRecord, load_jsonl, save_jsonl
from rca.systems import ALL, factories
from rca.systems.llm_systems import MultiAgent, MultiAgentVerified
from rca.taxonomy import LABELS, normalize_label


def rec(**kw):
    base = dict(id="x", project="p", test_name="T.test", log="", label="product_bug")
    base.update(kw)
    return FailureRecord(**base)


class TestTaxonomyAndSchema(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_label("Flaky"), "flaky")
        self.assertEqual(normalize_label("dependency-config"), "dependency_config")
        self.assertEqual(normalize_label("infra"), "environment")
        self.assertIsNone(normalize_label("banana"))
        self.assertIsNone(normalize_label(None))

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            recs = [rec(id="1", history=["P", "F"], meta={"k": 1}), rec(id="2", label=None)]
            save_jsonl(recs, p)
            back = load_jsonl(p)
            self.assertEqual([r.to_dict() for r in recs], [r.to_dict() for r in back])


class TestHeuristics(unittest.TestCase):
    def test_log_patterns(self):
        self.assertTrue(H.match_lines("java.io.IOException: No space left on device", "environment"))
        self.assertTrue(H.match_lines("ModuleNotFoundError: No module named 'x'", "dependency_config"))
        self.assertFalse(H.match_lines("all good", "environment"))

    def test_frame_kind(self):
        java_prod = "java.lang.NullPointerException\n\tat com.a.Foo.bar(Foo.java:1)\n\tat com.a.FooTest.testBar(FooTest.java:2)"
        java_test = "java.lang.AssertionError\n\tat com.a.FooTest.testBar(FooTest.java:2)\n\tat org.junit.runners.ParentRunner.run(ParentRunner.java:1)"
        py_prod = 'Traceback\n  File "tests/test_a.py", line 3, in test_x\n  File "src/a/core.py", line 9, in f\nKeyError'
        self.assertEqual(H.frame_kind(java_prod), "product")
        self.assertEqual(H.frame_kind(java_test), "test")
        self.assertEqual(H.frame_kind(py_prod), "product")
        self.assertIsNone(H.frame_kind("no frames here"))

    def test_history_signal(self):
        self.assertEqual(H.history_signal(list("PFPPF")), "flaky")
        self.assertEqual(H.history_signal(list("PPPP")), "regression")
        self.assertEqual(H.history_signal(list("FFFF")), "persistent")
        self.assertIsNone(H.history_signal(list("PF")))

    def test_commit_signal(self):
        self.assertEqual(H.commit_signal("Bump foo from 1.0 to 1.1", "")[0], "dependency_config")
        self.assertEqual(H.commit_signal("x", "+++ b/src/test/FooTest.java")[0], "test_bug")
        self.assertEqual(H.commit_signal("x", "+++ b/src/main/Foo.java")[0], "product_bug")
        self.assertIsNone(H.commit_signal("x", "+++ b/README.md")[0])


class TestMetrics(unittest.TestCase):
    def test_summary_known_values(self):
        gold = ["product_bug", "product_bug", "flaky", "flaky"]
        pred = ["product_bug", "flaky", "flaky", "flaky"]
        s = M.summary(gold, pred)
        self.assertAlmostEqual(s["accuracy"], 0.75)
        self.assertAlmostEqual(s["per_class"]["flaky"]["precision"], 2 / 3)
        self.assertAlmostEqual(s["per_class"]["flaky"]["recall"], 1.0)
        self.assertAlmostEqual(s["per_class"]["product_bug"]["recall"], 0.5)
        # macro is over classes present in gold (2 of 5)
        self.assertAlmostEqual(s["macro_recall"], 0.75)

    def test_invalid_prediction_counts_wrong(self):
        s = M.summary(["flaky"], ["invalid"])
        self.assertEqual(s["accuracy"], 0.0)
        self.assertEqual(M.confusion(["flaky"], ["invalid"])["flaky"]["invalid"], 1)

    def test_mcnemar(self):
        b, c, p = M.mcnemar_exact([1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1])
        self.assertEqual((b, c), (5, 1))
        self.assertAlmostEqual(p, 2 * (1 + 6) / 64)
        self.assertEqual(M.mcnemar_exact([1, 0], [1, 0])[2], 1.0)

    def test_kappa(self):
        a = ["x", "x", "y", "y"]
        self.assertAlmostEqual(M.cohen_kappa(a, a), 1.0)
        self.assertAlmostEqual(M.cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]), 0.0)

    def test_bootstrap_ci_brackets_mean(self):
        lo, hi = M.bootstrap_ci([1] * 70 + [0] * 30)
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.7)


class TestGrounding(unittest.TestCase):
    def test_split(self):
        r = rec(log="java.lang.NullPointerException: value is null\n  at A.b(A.java:1)", history=["P", "F"])
        ok, bad = split_grounding(r, ["NullPointerException:   value is null", "OutOfMemoryError", "P F", "...nullpointerexception..."])
        self.assertEqual(ok, ["NullPointerException:   value is null", "P F", "...nullpointerexception..."])
        self.assertEqual(bad, ["OutOfMemoryError"])


class TestPrompts(unittest.TestCase):
    def test_roundtrip_sections(self):
        r = rec(log="line1\nline2", history=["P", "F"], commit_message="msg", diff="+++ b/a")
        sec = parse_sections(render_case(r))
        self.assertEqual(sec["LOG"], "line1\nline2")
        self.assertEqual(sec["HISTORY"], "P F")
        self.assertEqual(sec["COMMIT"], "msg")

    def test_extract_json(self):
        self.assertEqual(extract_json('noise {"a": 1} tail'), {"a": 1})
        self.assertIsNone(extract_json("no json"))


class TestFolds(unittest.TestCase):
    def test_project_folds_have_no_leakage(self):
        recs = make_sample(80, seed=1)
        folds = make_folds(recs, k=4, seed=0)
        self.assertEqual(sorted(i for f in folds for i in f), list(range(80)))
        proj_fold = {}
        for fi, f in enumerate(folds):
            for i in f:
                self.assertEqual(proj_fold.setdefault(recs[i].project, fi), fi)

    def test_stratified_fallback_when_few_projects(self):
        recs = [rec(id=str(i), project="same", label=LABELS[i % 5]) for i in range(20)]
        folds = make_folds(recs, k=4)
        self.assertEqual(sorted(i for f in folds for i in f), list(range(20)))
        self.assertTrue(all(len(f) == 5 for f in folds))


class TestSystems(unittest.TestCase):
    def test_all_systems_run_on_sample(self):
        recs = make_sample(60, seed=0)
        rows = run_experiment(recs, factories(ALL, MockLLM()), k=3, seed=0)
        self.assertEqual(len(rows), 60 * len(ALL))
        for name in ALL:
            self.assertEqual(len([r for r in rows if r["system"] == name]), 60)
        self.assertTrue(all(r["pred"] in LABELS for r in rows))
        calls = {n: next(r["calls"] for r in rows if r["system"] == n) for n in ALL}
        self.assertEqual((calls["single"], calls["multi"], calls["multi_verifier"]), (1, 4, 5))

    def test_verifier_removes_fabricated_evidence(self):
        recs = make_sample(40, seed=2)
        llm = MockLLM(fabricate=1.0)
        plain = [MultiAgent(llm).diagnose(r) for r in recs]
        verified = [MultiAgentVerified(llm).diagnose(r) for r in recs]
        fake = "java.lang.FabricatedException: not in the input"
        self.assertTrue(all(fake in d.evidence for d in plain))
        self.assertFalse(any(fake in d.evidence for d in verified))
        self.assertTrue(all(split_grounding(r, d.evidence)[1] == [] for r, d in zip(recs, verified)))

    def test_hallucination_rate_reported(self):
        recs = make_sample(40, seed=3)
        rows = run_experiment(recs, factories(["single", "multi_verifier"], MockLLM(fabricate=1.0)), k=2)
        s = summarize(rows)["systems"]
        self.assertGreater(s["single"]["hallucination_rate"], 0.0)
        self.assertEqual(s["multi_verifier"]["hallucination_rate"], 0.0)

    def test_bad_llm_output_is_invalid_not_crash(self):
        class Garbage:
            is_mock = False

            def complete(self, role, system, user):
                from rca.llm import LLMResponse
                return LLMResponse("not json at all")

        recs = make_sample(10, seed=0)
        rows = run_experiment(recs, factories(["single"], Garbage()), k=2)
        self.assertTrue(all(r["pred"] == "invalid" and r["correct"] == 0 for r in rows))

    def test_cache_replays(self):
        with tempfile.TemporaryDirectory() as d:
            llm = CachedLLM(MockLLM(), d)
            r = make_sample(5)[0]
            a = llm.complete("single", "sys", render_case(r))
            b = llm.complete("single", "sys", render_case(r))
            self.assertEqual((llm.hits, llm.misses), (1, 1))
            self.assertEqual(a.text, b.text)

    def test_report_written(self):
        recs = make_sample(40, seed=0)
        rows = run_experiment(recs, factories(["rule", "tfidf"], MockLLM()), k=2)
        with tempfile.TemporaryDirectory() as d:
            path = write_report(summarize(rows), d, {"n_records": 40, "folds": 2, "seed": 0, "llm": "mock", "mock": True, "data": "x"})
            text = path.read_text()
            self.assertIn("OFFLINE / VALIDATION WARNING", text)
            self.assertIn("| rule |", text)
            self.assertTrue((Path(d) / "summary.json").exists())


class TestConverters(unittest.TestCase):
    def test_idoft(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pr-data.csv"
            p.write_text("Project URL,SHA Detected,Module Path,Fully-Qualified Test Name (packageName.ClassName.methodName),Category,Status,PR Link,Notes\n"
                         "https://github.com/foo/bar,abc,m,pkg.Cls.test1,OD,Opened,http://pr,\n"
                         ",,,,,,,\n")
            recs = convert_idoft(p)
            self.assertEqual(len(recs), 1)
            self.assertEqual((recs[0].label, recs[0].project, recs[0].meta["category"]), ("flaky", "foo/bar", "OD"))

    def test_bugswarm(self):
        self.assertEqual(weak_label({"code": "Yes", "test": "No", "build": "No"}), "product_bug")
        self.assertEqual(weak_label({"code": "No", "test": "Yes", "build": "No"}), "test_bug")
        self.assertEqual(weak_label({"code": "No", "test": "No", "build": "Yes"}), "dependency_config")
        self.assertIsNone(weak_label({}))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "logs").mkdir()
            (Path(d) / "logs" / "a-1.log").write_text("boom")
            j = Path(d) / "x.json"
            j.write_text(json.dumps([{"image_tag": "a-1", "repo": "o/r", "lang": "Java", "failed_job": {"job_id": 7},
                                      "classification": {"code": "Yes", "test": "No", "build": "No"}}, {"image_tag": "b-2"}]))
            recs = convert_bugswarm(j, Path(d) / "logs")
            self.assertEqual((recs[0].log, recs[0].label, recs[0].label_source), ("boom", "product_bug", "bugswarm-weak"))
            self.assertIsNone(recs[1].label)

    def test_bugswarm_generated_artifact_signal(self):
        diff = """--- failed/sqlmodel/sqlmodel/sql/_expression_select_gen.py
+++ passed/sqlmodel/sqlmodel/sql/_expression_select_gen.py
@@ -1,6 +1,5 @@
 # WARNING: do not modify this code, it is generated by _expression_select_gen.py.jinja2

-
 from datetime import datetime
"""

        from rca import heuristics as H

        files = H.changed_files(diff)
        self.assertEqual(
            files,
            ["sqlmodel/sqlmodel/sql/_expression_select_gen.py"],
        )

        signal, evidence = H.commit_signal(
            "🎨 [pre-commit.ci] Auto format from pre-commit.com hooks",
            diff,
        )

        self.assertEqual(signal, "product_bug")
        self.assertEqual(
            evidence,
            [
                "+++ passed/sqlmodel/sqlmodel/sql/_expression_select_gen.py",
            ],
        )

    def test_dir_loader(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "projA").mkdir()
            (root / "projA" / "f1.log").write_text("No space left on device")
            (root / "projA" / "f1.commit.txt").write_text("msg")
            (root / "labels.csv").write_text("id,label\nprojA/f1,environment\n")
            recs = convert_dir(root, root / "labels.csv", "gitbug")
            self.assertEqual(len(recs), 1)
            self.assertEqual((recs[0].project, recs[0].label, recs[0].commit_message), ("projA", "environment", "msg"))


class TestLabeling(unittest.TestCase):
    def test_sheet_roundtrip_and_kappa(self):
        recs = make_sample(10)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.csv", Path(d) / "b.csv"
            labeling.export_sheet(recs, a)
            labeling.export_sheet(recs, b)
            import csv

            def fill(path, labels):
                with path.open() as fh:
                    rows = list(csv.DictReader(fh))
                for row, lab in zip(rows, labels):
                    row["label"] = lab
                with path.open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=labeling.SHEET_COLUMNS)
                    w.writeheader()
                    w.writerows(rows)

            gold = [r.label for r in recs]
            fill(a, gold)
            fill(b, gold[:-1] + ["flaky" if gold[-1] != "flaky" else "test_bug"])
            res = labeling.kappa_from_sheets(a, b)
            self.assertEqual(res["n"], 10)
            self.assertEqual(len(res["disagreements"]), 1)
            blank = [rec(id=r.id, label=None) for r in recs]
            applied, unknown = labeling.apply_sheet(blank, a)
            self.assertEqual((applied, unknown), (10, []))
            self.assertEqual([r.label for r in blank], gold)
            self.assertTrue(all(r.label_source == "manual" for r in blank))


if __name__ == "__main__":
    unittest.main()
