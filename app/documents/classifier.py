"""Deterministic keyword classifier used by the mock provider and as a cross-check."""

import re

from app.models.common import DocumentType

_SIGNALS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PAY_STUB: (
        r"earnings statement",
        r"pay statement",
        r"pay stub",
        r"gross (pay|earnings)",
        r"net pay",
        r"pay period",
        r"year-to-date|ytd",
    ),
    DocumentType.EMPLOYMENT_LETTER: (
        r"to whom it may concern",
        r"employment verification",
        r"confirms? that",
        r"has been employed",
        r"sincerely",
    ),
    DocumentType.BANK_STATEMENT: (
        r"account holder",
        r"transaction (summary|history)",
        r"statement period",
        r"opening balance|closing balance",
        r"account:? \*+\d+",
    ),
}

MIN_SIGNALS = 2


def classify(text: str) -> tuple[DocumentType, dict[str, int]]:
    """Return the best-scoring type, or UNKNOWN when no type has enough signals."""
    lowered = text.lower()
    scores = {
        doc_type.value: sum(1 for pattern in patterns if re.search(pattern, lowered))
        for doc_type, patterns in _SIGNALS.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    tied = [t for t, s in scores.items() if s == best_score]
    if best_score < MIN_SIGNALS or len(tied) > 1:
        return DocumentType.UNKNOWN, scores
    return DocumentType(best_type), scores
