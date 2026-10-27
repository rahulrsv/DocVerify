"""Extraction provider interface. Providers return raw facts; they never decide outcomes."""

from dataclasses import dataclass, field
from typing import Protocol

from app.models.extraction import RawExtraction


class ProviderError(Exception):
    def __init__(self, kind: str, message: str, attempts: int = 1) -> None:
        super().__init__(message)
        self.kind = kind  # refusal | malformed | timeout | api_error | auth | incomplete
        self.attempts = attempts


@dataclass
class ProviderResult:
    raw: RawExtraction
    attempts: int = 1
    latency_ms: int = 0
    notes: list[str] = field(default_factory=list)


class ExtractionProvider(Protocol):
    name: str
    model: str
    prompt_version: str

    def extract(self, text: str, document_id: str) -> ProviderResult:
        """Extract raw fields from untrusted document text. Raise ``ProviderError`` on failure."""
        ...
