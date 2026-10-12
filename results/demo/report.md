# Root-cause analysis: results

> **MOCK LLM / SYNTHETIC DATA WARNING.** If this run used `--llm mock` or the bundled synthetic sample, the numbers below only show that the pipeline works. They are not research results.

- Records: **150**, folds: **5** (grouped by project), seed 0
- LLM: `mock`  |  data: `data/sample.jsonl`

## Overall

| System | Accuracy [95% CI] | Macro-F1 | Hallucination | Mean latency (s) | Mean cost ($) | Calls/record |
|---|---|---|---|---|---|---|
| rule | 72.7% [65.3%, 80.0%] | 0.753 | 0.0% | 0.000 | 0.00000 | 0.0 |
| tfidf | 100.0% [100.0%, 100.0%] | 1.000 | n/a | 0.000 | 0.00000 | 0.0 |
| single | 90.0% [85.3%, 94.7%] | 0.895 | 0.0% | 0.000 | 0.00000 | 1.0 |
| rag | 96.0% [92.7%, 98.7%] | 0.959 | 0.0% | 0.000 | 0.00000 | 1.0 |
| multi | 80.0% [74.0%, 86.0%] | 0.808 | 0.0% | 0.000 | 0.00000 | 4.0 |
| multi_verifier | 84.0% [78.0%, 89.3%] | 0.835 | 0.0% | 0.000 | 0.00000 | 5.0 |

Hallucination = share of cited evidence items that do not occur verbatim in the input (n/a for systems that cite no evidence). Cost is 0 unless `RCA_PRICE_IN_PER_MTOK` / `RCA_PRICE_OUT_PER_MTOK` are set.

## Paired significance vs `rule` (exact McNemar)

| System | Only this right | Only reference right | p-value |
|---|---|---|---|
| tfidf | 41 | 0 | 0.0000 |
| single | 31 | 5 | 0.0000 |
| rag | 37 | 2 | 0.0000 |
| multi | 11 | 0 | 0.0010 |
| multi_verifier | 20 | 3 | 0.0005 |

## Per-class F1

| System | product_bug | test_bug | flaky | environment | dependency_config |
|---|---|---|---|---|---|
| rule | 0.594 | 0.750 | 0.824 | 0.824 | 0.776 |
| tfidf | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| single | 0.909 | 0.983 | 0.870 | 0.750 | 0.966 |
| rag | 0.984 | 1.000 | 0.923 | 0.889 | 1.000 |
| multi | 0.690 | 0.750 | 0.778 | 0.824 | 1.000 |
| multi_verifier | 0.779 | 0.750 | 0.896 | 0.750 | 1.000 |

Support: product_bug=30, test_bug=30, flaky=30, environment=30, dependency_config=30

## Confusion matrices (rows = gold, columns = predicted)

### rule

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 12 | 18 | 0 | 0 | 0 | 0 |
| flaky | 9 | 0 | 21 | 0 | 0 | 0 |
| environment | 9 | 0 | 0 | 21 | 0 | 0 |
| dependency_config | 11 | 0 | 0 | 0 | 19 | 0 |

### tfidf

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 0 | 30 | 0 | 0 | 0 | 0 |
| flaky | 0 | 0 | 30 | 0 | 0 | 0 |
| environment | 0 | 0 | 0 | 30 | 0 | 0 |
| dependency_config | 0 | 0 | 0 | 0 | 30 | 0 |

### single

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 0 | 29 | 1 | 0 | 0 | 0 |
| flaky | 0 | 0 | 30 | 0 | 0 | 0 |
| environment | 6 | 0 | 6 | 18 | 0 | 0 |
| dependency_config | 0 | 0 | 2 | 0 | 28 | 0 |

### rag

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 0 | 30 | 0 | 0 | 0 | 0 |
| flaky | 0 | 0 | 30 | 0 | 0 | 0 |
| environment | 1 | 0 | 5 | 24 | 0 | 0 |
| dependency_config | 0 | 0 | 0 | 0 | 30 | 0 |

### multi

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 12 | 18 | 0 | 0 | 0 | 0 |
| flaky | 9 | 0 | 21 | 0 | 0 | 0 |
| environment | 6 | 0 | 3 | 21 | 0 | 0 |
| dependency_config | 0 | 0 | 0 | 0 | 30 | 0 |

### multi_verifier

| gold \ pred | product_bug | test_bug | flaky | environment | dependency_config | invalid |
|---|---|---|---|---|---|---|
| product_bug | 30 | 0 | 0 | 0 | 0 | 0 |
| test_bug | 11 | 18 | 1 | 0 | 0 | 0 |
| flaky | 0 | 0 | 30 | 0 | 0 | 0 |
| environment | 6 | 0 | 6 | 18 | 0 | 0 |
| dependency_config | 0 | 0 | 0 | 0 | 30 | 0 |
