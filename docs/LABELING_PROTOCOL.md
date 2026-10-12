# Labelling protocol (v0.1: refine after your pilot round)

Goal: assign each failure the **primary root cause**: the thing that, if fixed, makes the failure go away.

## Categories and decision order

Apply the questions in order and stop at the first "yes".

1. **environment**: Is the failure caused by infrastructure rather than the code under test (disk full, out of memory, network or DNS errors, runner or container problems, an unavailable external service)? Would rerunning on healthy infrastructure pass with no code change?
2. **dependency_config**: Is it caused by a dependency, build or configuration problem (missing or incompatible package, wrong version, resolution failure, missing config or environment variable, unsupported runtime version)? Often follows a version bump or a build-file change.
3. **flaky**: Is the outcome non-deterministic with no relevant code change (timing, races, ordering, randomness, shared state between tests)? Passes on rerun, or the history mixes passes and failures.
4. **test_bug**: Is the production code behaving correctly while the test is wrong (stale expectation, hard-coded data, bad mock or fixture, over-strict assertion, missing setup)? The fix would be in the test.
5. **product_bug**: Otherwise: the production code is wrong and the test correctly exposes it. The fix would be in production code.

## Rules

- Decide from the **fix** where one exists (BugSwarm/GitBug-Java provide it): which file would you change? Use the log only when no fix is available.
- One label per failure. If two causes seem equally primary, choose by the order above and write both in `notes`.
- Do not label from the model's output. Label before you look at any system's prediction.
- Write down every ambiguous case in `notes`; they become the basis for refining this protocol.

## Process

1. Both annotators label the same 20-30% sample independently (`rca label-sheet` -> fill `label`).
2. Run `rca kappa a.csv b.csv`. Discuss every disagreement, update this document, relabel if needed.
3. Target Cohen's kappa of about 0.7 or higher before one annotator labels the rest. Report the final kappa and the sample size.
4. Keep `label_source` as `manual`; keep weak labels (`bugswarm-weak`) out of the evaluation set unless confirmed.
