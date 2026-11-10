"""TF-IDF retriever over the fictional knowledge base."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
        self.m = self.vec.transform(docs)

    def top(self, query: str, k: int = 2) -> list[str]:
        sims = linear_kernel(self.vec.transform([query]), self.m)[0]
        return [self.docs[i] for i in sims.argsort()[::-1][:k]]
