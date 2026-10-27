"""Deterministic prompt-injection signal scanner.

This is one defensive layer, not a guarantee. The primary controls are structural:
document text is sent as delimited untrusted data, the extraction schema cannot express
a decision, extracted values must be grounded in the source text, and the policy engine
never consumes model recommendations.
"""

import re

_PATTERNS: tuple[tuple[str, str], ...] = (
    ("IGNORE_INSTRUCTIONS", r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules?|prompts?)\b"),
    ("SYSTEM_OVERRIDE", r"\bsystem\s*(override|prompt|message|instruction)s?\b"),
    ("ROLE_IMPERSONATION", r"(^|\n)\s*(assistant|system|ai|model|chatgpt|llm)\s*:"),
    ("ADDRESSES_AI", r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|assistant|model|llm|language model|extraction system)\b"),
    ("DECISION_MANIPULATION", r"\b(mark|set|report|output|flag|classify)\b.{0,40}\b(approved?|approval|decision)\b"),
    ("FORCED_OUTPUT", r"\byou (must|should|will) (output|return|report|respond|extract)\b"),
)


def scan_for_injection(text: str) -> list[str]:
    """Return the distinct signal codes found in ``text``."""
    lowered = text.lower()
    return [code for code, pattern in _PATTERNS if re.search(pattern, lowered, flags=re.IGNORECASE)]
