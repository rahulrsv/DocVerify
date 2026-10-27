"""Okapi BM25 over handbook passages. Pure Python, deterministic, no model download."""

import math
import re
from collections import Counter

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "cannot", "do", "does", "for", "from", "has",
        "have", "how", "if", "in", "into", "is", "it", "its", "may", "must", "no", "not", "of", "on", "or", "should",
        "so", "such", "than", "that", "the", "their", "then", "there", "these", "this", "to", "was", "were", "what",
        "when", "which", "who", "why", "will", "with",
    }
)


def _stem(token: str) -> str:
    # Deliberately crude suffix stripping: enough to match "deposits"/"deposit" and "dated"/"date".
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and not t.isdigit()]


class BM25Scorer:
    name = "bm25"

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._docs = [Counter(tokenize(d)) for d in documents]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        df = Counter(term for doc in self._docs for term in doc)
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self._docs, self._lengths):
            norm = self._k1 * (1 - self._b + self._b * length / self._avg_len)
            out.append(sum(self._idf[t] * doc[t] * (self._k1 + 1) / (doc[t] + norm) for t in terms if t in doc))
        return out
