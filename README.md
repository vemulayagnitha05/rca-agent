# rca-agent: root-cause analysis of test failures, rule-based vs ML vs LLM agents

An evaluation framework for one research question:

> Can a multi-agent LLM system with a verifier identify the root cause of software test failures more accurately,
> and with fewer fabricated explanations, than a rule-based baseline, a trained ML classifier, a single LLM or a RAG-augmented LLM?

It is a *measurement* project, not a product. The value is in the dataset, the labelling protocol and the evaluation, so those are
built in: leakage-aware cross-validation, manual-labelling tooling with inter-annotator agreement, bootstrap confidence intervals, paired
significance tests, and an evidence-grounding check that turns "hallucination" into a number.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://rca-agent-demo.streamlit.app/)
[![Tests](https://github.com/vemulayagnitha05/rca-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/vemulayagnitha05/rca-agent/actions/workflows/tests.yml)

## 🚀 Live Demo

**Try the deployed Streamlit app:**  
https://rca-agent-demo.streamlit.app/

The live demo provides an interactive rule-based root-cause analyzer for software test failures. Paste a failing test log and receive a predicted root-cause category, confidence score, hypothesis, and supporting evidence.

> **Demo note:** The deployed app is an interactive demonstration of the rule-based baseline. It does not use an external LLM and should not be interpreted as a benchmark or research result.

![RCA Agent Streamlit Demo](screenshots/demo-diagnosis.png)

### Highlights

- Root-cause taxonomy covering product_bug, test_bug, flaky, environment, and dependency_config
- Comparison framework for rule-based, TF-IDF/ML, single-agent, RAG, multi-agent, and verifier-based approaches
- Leakage-aware cross-validation and automated evaluation metrics
- Bootstrap confidence intervals and paired significance testing
- Manual labeling tooling with inter-annotator agreement
- Evidence-grounding checks for unsupported explanations
- 26 automated tests with GitHub Actions CI
- Deployed Streamlit demonstration for interactive RCA

## Status: what is verified and what is not

| Part | Status |
|---|---|
| Harness, metrics, systems, converters, labelling tools | Written and covered by 26 unit tests (all pass). |
| Offline demo on synthetic data with the mock LLM | Runs end to end. **Pipeline check only, not a research result.** |
| Real LLM runs (`--llm anthropic`) | Implemented, **not run** (no API access or datasets in the build environment). |
| IDoFT, BugSwarm, GitBug-Java, LogChunks | **Not downloaded** (network blocked in the build environment). Converters are written against the documented formats; the BugSwarm field names in particular were not checked against a live export. Test the converters on a few real rows first. |

The mock LLM (`--llm mock`) is a deterministic heuristic that reads the same signals as the rule-based system. Differences between systems under
the mock reflect how those heuristics combine, not how language models behave. For the same reason the TF-IDF baseline scores 100% on the synthetic
sample: the logs are template-generated. Never report numbers from the synthetic sample.

## Systems compared

| Name | What it is | LLM calls/record |
|---|---|---|
| `rule` | Regex signatures on the log, then "where was the exception thrown" (test vs production code). No history, no commit. | 0 |
| `tfidf` | TF-IDF + logistic regression on test name, log and commit message, trained inside each fold. The *trained ML* baseline. | 0 |
| `single` | One prompt that sees the full case (log, history, commit, diff). A fair monolithic baseline. | 1 |
| `rag` | `single` plus the 3 most similar labelled failures from the training fold. | 1 |
| `multi` | Log analyst + history analyst + change analyst, then a lead investigator. | 4 |
| `multi_verifier` | `multi`, then (1) a deterministic grounding check that drops evidence not in the input, (2) a skeptical LLM reviewer that may revise the label. | 5 |

Root-cause taxonomy (edit `src/rca/taxonomy.py`): `product_bug`, `test_bug`, `flaky`, `environment`, `dependency_config`.

## Quick start (offline demo)

```bash
pip install -e .            # needs scikit-learn
python -m rca make-sample --n 150 --out data/sample.jsonl     # synthetic, for pipeline testing only
python -m rca run --data data/sample.jsonl --folds 5 --out results/demo
python -m unittest discover -s tests                          # or: pytest
```

`results/demo/` in this repository is the output of exactly that command, kept as an example of the report format.

## Running the real experiment

### 1. Get the data (on your own machine)

| Dataset | Where | What you get | Converter |
|---|---|---|---|
| BugSwarm | bugswarm.org, github.com/bugswarm/bugswarm (REST API, CLI) | Failing/passing CI pairs, Java and Python; reproducible containers | `rca convert-bugswarm --json export.json --logs-dir logs/ --out data/bugswarm.jsonl` |
| GitBug-Java | github.com/gitbugactions/gitbug-java (setup needs about 130 GB: ~50 GiB image + ~80 GiB dependencies) | 199 recent (2023) Java bugs from 55 repos; use as a held-out set against model memorisation | collect logs with `gitbug-java run`, then `rca convert-dir` |
| IDoFT | github.com/TestingResearchIllinois/idoft | Flaky tests with categories (OD, ID, NOD, ...); positives only, no logs | `rca convert-idoft --csv pr-data.csv --out data/idoft.jsonl` |
| LogChunks | Zenodo, DOI 10.5281/zenodo.3632351 (CC BY 4.0 on the Hugging Face copy) | 797 Travis build logs from 80 repos with the labelled failure chunk | `rca convert-dir --dir logchunks/logs --source logchunks --out ...` |

Check each dataset's licence before redistributing anything derived from it.

### 2. Label (this is where the thesis credibility comes from)

None of these datasets gives you labels in this taxonomy. BugSwarm's `classification` records what the *fix* touched, so the converter maps it to
**weak** labels only, and `flaky`/`environment` cannot be derived. Read `docs/LABELING_PROTOCOL.md`, then:

```bash
python -m rca label-sheet data/bugswarm.jsonl --out sheets/a.csv        # annotator A fills the `label` column
python -m rca label-sheet data/bugswarm.jsonl --out sheets/b.csv        # annotator B labels a 20-30% sample
python -m rca kappa sheets/a.csv sheets/b.csv                           # inter-annotator agreement
python -m rca apply-labels --data data/bugswarm.jsonl --sheet sheets/a.csv --out data/bugswarm.labeled.jsonl
```

Aim for 200-500 labelled failures with a second annotator on a sample. Report Cohen's kappa in the thesis.

### 3. Run

```bash
export ANTHROPIC_API_KEY=...
export RCA_MODEL=claude-sonnet-5-5                  # default; any model id works
export RCA_PRICE_IN_PER_MTOK=...  RCA_PRICE_OUT_PER_MTOK=...   # USD, from Anthropic's pricing page; otherwise cost shows 0
pip install -e ".[llm]"

python -m rca run --data data/bugswarm.labeled.jsonl data/gitbug.labeled.jsonl \
    --llm anthropic --limit 40 --workers 4 --out results/pilot      # always pilot small first
```

The CLI prints the estimated number of LLM calls and aborts above `--max-calls` (default 3000). Responses are cached on disk
(`<out>/llm_cache`), so a rerun costs nothing and gives identical numbers, which also makes the experiment reproducible.

Outputs in `--out`: `report.md` (tables), `summary.json`, `predictions.jsonl` (one row per system and record: prediction, evidence, tokens,
latency, cost, verifier notes, useful for error analysis).

## Metrics

- **Accuracy** with a 95% bootstrap CI; **macro precision / recall / F1** (macro over classes that occur in the gold labels), per-class F1, confusion matrices.
- **Paired significance**: exact McNemar test of each system against a reference (`--reference`, default `rule`) on the same records.
- **Hallucination rate**: share of cited evidence items that do not appear verbatim (whitespace/case-insensitive) in the input. It measures fabricated or
  misquoted evidence, not wrong labels. Systems that cite no evidence (`tfidf`) report n/a. Because the verifier filters ungrounded evidence by design, its
  rate is expected to be near zero; the interesting comparison is `multi` vs `multi_verifier` on **accuracy** and on how often the verifier changes the label.
- **Cost and latency**: mean per record, summed over all LLM calls of a system (calls within one system run sequentially, so latency is end to end).
- Invalid or unparseable model output counts as a wrong answer (`invalid`), and exceptions as `error`; both are shown in the confusion matrices.

## Experimental design notes

- **Project-grouped folds** (default): a project's failures never appear in both train and test, which avoids inflated scores from near-duplicate failures. Disable with `--no-group-by-project`.
- **Leakage**: Defects4J-style and older GitHub data are likely in LLM training data. Evaluate GitBug-Java (2023 bugs) separately and compare; a large gap suggests memorisation.
- **Information parity**: `single` and `rag` see exactly what the multi-agent system sees, so any gain is attributable to the architecture and not to extra information.
- **Prompts are fixed in `src/rca/prompts.py`**, shared by all LLM systems. Tune them on a separate development split, not on the evaluation folds.
- **History** is the prior pass/fail outcomes of the failing test (oldest to newest, current failure excluded). Real datasets rarely ship it; BugSwarm/CI data
  can provide it by mining earlier runs. Without history, the history analyst contributes nothing: report that as a limitation or mine it.

## Threats to validity (put these in the write-up)

1. Label quality: weak labels, a single annotator, or low kappa undermine every number.
2. Class imbalance: real data is dominated by a few classes; read per-class F1, not only accuracy.
3. Model memorisation of public benchmarks.
4. Prompt sensitivity: report the prompt version and, ideally, repeat with a second phrasing.
5. Non-determinism: temperature 0 helps but does not guarantee identical outputs across API versions; the cache pins them for reproduction.
6. Taxonomy fit: some failures have several causes; the protocol picks the primary one.

## Run the Streamlit demo locally

````powershell
streamlit run app.py
````

The demo uses the rule-based baseline and reads the test name and failing test log.

## Layout

```
src/rca/
  taxonomy.py  schema.py          labels, FailureRecord (canonical JSONL)
  datasets/                       idoft, bugswarm, dirloader converters + synthetic sample
  systems/                        rule, tfidf, retrieval, llm_systems (single, rag, multi, verifier)
  llm.py  prompts.py              Anthropic client, disk cache, mock LLM, prompts
  grounding.py  heuristics.py     evidence-grounding check, hand-written signals
  evaluate.py  metrics.py  report.py  labeling.py  cli.py
tests/test_core.py                26 unit tests
data/sample.jsonl                 synthetic sample (pipeline testing only)
results/demo/                     example report from the offline demo
docs/LABELING_PROTOCOL.md
```

## Extending

- New system: subclass `System` in `systems/`, implement `diagnose`, register it in `systems/__init__.py`.
- New label set: edit `taxonomy.py`; prompts, metrics and reports follow.
- Self-healing follow-up (project idea #2): reuse the diagnosis as input to a patch-generation agent and verify by re-running the failing test in the reproducible container BugSwarm / GitBug-Java provide.
