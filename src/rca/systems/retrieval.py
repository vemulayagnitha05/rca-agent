from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..textutil import case_text


class Retriever:
    """TF-IDF nearest-neighbour search over labelled training failures (used by the RAG system)."""

    def __init__(self):
        self.vec = None
        self.X = None
        self.records = []

    def fit(self, records) -> None:
        self.records = [r for r in records if r.label]
        if not self.records:
            return
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        self.X = self.vec.fit_transform([case_text(r) for r in self.records])

    def query(self, rec, k: int = 3):
        if self.vec is None:
            return []
        sims = linear_kernel(self.vec.transform([case_text(rec)]), self.X).ravel()
        order = sims.argsort()[::-1][:k]
        return [self.records[int(i)] for i in order]
