"""Per-query router. A lightweight classifier predicts P(correct | query, strategy) for each strategy
from query text only; the router picks argmax_s P(s) - lam * cost(s). Evaluation uses out-of-fold
predictions (k-fold CV), so no query is routed by a model that saw its own label."""
from __future__ import annotations

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def handcrafted(q: str) -> list[float]:
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    return [len(q) / 300, len(q.split()) / 60, len(nums) / 8, q.count("?"), q.count(","), float("$" in q),
            float(bool(re.match(r"^what is \d+ [-+*] \d+\?$", q.strip().lower()))),
            float(bool(re.search(r"\b(who|capital|founded)\b", q.lower())))]


def featurize(train_q: list[str], test_q: list[str]):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True).fit(train_q)
    f = lambda qs: hstack([vec.transform(qs), csr_matrix(np.array([handcrafted(q) for q in qs]))]).tocsr()
    return f(train_q), f(test_q)


def oof_success_probs(questions: list[str], correct: np.ndarray, families: list[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """correct: [n_queries, n_strategies] bool -> out-of-fold P(correct), same shape. Folds stratified by family."""
    n, S = correct.shape
    P = np.zeros((n, S))
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), families):
        Xtr, Xte = featurize([questions[i] for i in tr], [questions[i] for i in te])
        for s in range(S):
            y = correct[tr, s]
            if y.all() or not y.any():
                P[te, s] = float(y.mean())
                continue
            P[te, s] = LogisticRegression(C=2.0, max_iter=2000).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return P


def route(P: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    """cost: [S] or [n, S], normalized. Returns the chosen strategy index per query."""
    return np.argmax(P - lam * cost, axis=1)


def evaluate(choice: np.ndarray, correct: np.ndarray, flops: np.ndarray) -> tuple[float, float]:
    idx = np.arange(len(choice))
    return float(correct[idx, choice].mean()), float(flops[idx, choice].mean())


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points: (cost, accuracy). Non-dominated subset sorted by cost."""
    front, best = [], -1.0
    for c, a in sorted(set(points), key=lambda p: (p[0], -p[1])):
        if a > best:
            front.append((c, a))
            best = a
    return front
