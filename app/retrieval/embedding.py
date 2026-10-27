"""Optional dense-vector scorer. Any encoder works; sentence-transformers is the free local default.

Nothing here is imported unless DOCVERIFY_RETRIEVAL_BACKEND=embedding, so the base install
and the test suite stay offline and dependency-free.
"""

import math
from typing import Protocol


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Runs a small local model (default all-MiniLM-L6-v2, ~90 MB, CPU). No API calls."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("embedding retrieval needs `pip install -r requirements-embeddings.txt`") from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class EmbeddingScorer:
    name = "embedding"

    def __init__(self, documents: list[str], encoder: Encoder) -> None:
        self.encoder = encoder
        self._vectors = [_unit(v) for v in encoder.encode(documents)]

    def scores(self, query: str) -> list[float]:
        q = _unit(self.encoder.encode([query])[0])
        return [sum(a * b for a, b in zip(q, v)) for v in self._vectors]
