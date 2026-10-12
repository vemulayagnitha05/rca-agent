"""Baseline 2: a trained ML classifier (TF-IDF + logistic regression) on log, test name and commit text."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..textutil import case_text
from .base import Diagnosis, System


class TfidfClassifier(System):
    name = "tfidf_lr"

    def __init__(self):
        self.vec = None
        self.clf = None
        self.constant: str | None = None

    def fit(self, train) -> None:
        train = [r for r in train if r.label]
        labels = [r.label for r in train]
        if len(set(labels)) < 2:
            self.constant = labels[0] if labels else "product_bug"
            return
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform([case_text(r) for r in train])
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.clf.fit(X, labels)

    def diagnose(self, rec) -> Diagnosis:
        if self.constant is not None:
            return Diagnosis(self.constant, "Training data had a single class.", [], 0.0)
        X = self.vec.transform([case_text(rec)])
        probs = self.clf.predict_proba(X)[0]
        i = int(probs.argmax())
        return Diagnosis(str(self.clf.classes_[i]), "", [], float(probs[i]))
