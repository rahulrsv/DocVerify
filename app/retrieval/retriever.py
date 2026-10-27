"""Policy-text retrieval: finds the handbook passage that explains a rule result or answers a question.

Retrieval only annotates explanations. It runs after the decision is made and never feeds
back into rules or outcomes.
"""

from dataclasses import dataclass
from typing import Protocol

from app.models.case import PolicyCitation, RuleResult
from app.models.common import RuleOutcome
from app.retrieval.corpus import Handbook, Passage

EXCERPT_CHARS = 280


class Scorer(Protocol):
    name: str

    def scores(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float
    rank: int


class PolicyRetriever:
    def __init__(self, handbook: Handbook, scorer: Scorer) -> None:
        self.handbook = handbook
        self.scorer = scorer

    @property
    def version(self) -> str:
        encoder = getattr(self.scorer, "encoder", None)
        return f"{self.handbook.version}/{self.scorer.name}" + (f":{encoder.name}" if encoder else "")

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        scored = sorted(zip(self.handbook.passages, self.scorer.scores(query)), key=lambda item: (-item[1], item[0].passage_id))
        return [SearchHit(p, round(s, 4), i + 1) for i, (p, s) in enumerate(scored[:k]) if s > 0]

    def cite(self, result: RuleResult) -> PolicyCitation | None:
        """Retrieve from the rule's title and free-text explanation, not its ID, so a miss is visible."""
        hits = self.search(f"{result.title}. {result.explanation}", k=1)
        if not hits:
            return None
        hit = hits[0]
        excerpt = hit.passage.text if len(hit.passage.text) <= EXCERPT_CHARS else hit.passage.text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "..."
        return PolicyCitation(
            passage_id=hit.passage.passage_id,
            title=hit.passage.title,
            excerpt=excerpt,
            score=hit.score,
            retriever=self.version,
            matches_rule=hit.passage.rule_id == result.rule_id,
        )

    def annotate(self, results: list[RuleResult]) -> list[RuleResult]:
        return [r.model_copy(update={"citation": self.cite(r)}) if r.outcome is RuleOutcome.FAIL else r for r in results]
