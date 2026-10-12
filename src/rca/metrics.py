"""Classification metrics, bootstrap confidence intervals, McNemar test and Cohen's kappa. Stdlib only."""

from __future__ import annotations

import math
import random
from collections import Counter

from .taxonomy import LABELS


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """matrix[gold][pred]; predictions outside the taxonomy are counted under 'invalid'."""
    cols = LABELS + ["invalid"]
    m = {g: {p: 0 for p in cols} for g in LABELS}
    for g, p in zip(gold, pred):
        if g in m:
            m[g][p if p in LABELS else "invalid"] += 1
    return m


def per_class(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for g in gold if g == lab)}
    return out


def summary(gold: list[str], pred: list[str]) -> dict:
    pc = per_class(gold, pred)
    present = [lab for lab in LABELS if pc[lab]["support"] > 0]  # macro over classes that exist in the gold set
    n = len(gold)
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0,
        "macro_precision": sum(pc[lab]["precision"] for lab in present) / len(present) if present else 0.0,
        "macro_recall": sum(pc[lab]["recall"] for lab in present) / len(present) if present else 0.0,
        "macro_f1": sum(pc[lab]["f1"] for lab in present) / len(present) if present else 0.0,
        "per_class": pc,
    }


def bootstrap_ci(correct: list[int], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy from a 0/1 correctness vector."""
    if not correct:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    means = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def mcnemar_exact(correct_a: list[int], correct_b: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired correctness. Returns (b, c, p): b = A right & B wrong, c = reverse."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, p)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Inter-annotator agreement. 1 = perfect, 0 = chance level."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("label lists must be non-empty and the same length")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
