"""Builds the configured retriever. "bm25" (default) is offline; "embedding" runs a local model."""

from app.core.config import Settings
from app.policies.config import PolicyConfig
from app.retrieval.corpus import load_handbook
from app.retrieval.embedding import EmbeddingScorer, Encoder, SentenceTransformerEncoder
from app.retrieval.lexical import BM25Scorer
from app.retrieval.retriever import PolicyRetriever


def build_retriever(settings: Settings, policy: PolicyConfig, encoder: Encoder | None = None) -> PolicyRetriever | None:
    backend = settings.retrieval_backend
    if backend == "none":
        return None
    handbook = load_handbook(settings.handbook_path, policy)
    documents = [p.searchable for p in handbook.passages]
    if backend == "bm25":
        return PolicyRetriever(handbook, BM25Scorer(documents))
    if backend == "embedding":
        return PolicyRetriever(handbook, EmbeddingScorer(documents, encoder or SentenceTransformerEncoder(settings.embedding_model)))
    raise ValueError(f"unknown DOCVERIFY_RETRIEVAL_BACKEND {backend!r} (expected bm25, embedding or none)")
