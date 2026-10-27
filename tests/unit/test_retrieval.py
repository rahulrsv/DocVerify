import dataclasses
import re

import pytest

from app.core.config import Settings
from app.evals import retrieval as retrieval_eval
from app.models.case import RuleResult
from app.models.common import Outcome, RuleOutcome, Severity
from app.policies.rules import RULES
from app.retrieval.corpus import load_handbook, parse_handbook
from app.retrieval.embedding import EmbeddingScorer
from app.retrieval.factory import build_retriever
from app.retrieval.lexical import BM25Scorer, tokenize
from app.retrieval.retriever import PolicyRetriever


@pytest.fixture
def handbook(policy):
    return load_handbook(Settings().handbook_path, policy)


@pytest.fixture
def retriever(policy):
    return build_retriever(dataclasses.replace(Settings(), retrieval_backend="bm25"), policy)


def failed(rule_id: str, title: str, explanation: str) -> RuleResult:
    return RuleResult(rule_id=rule_id, rule_version="1.0", title=title, outcome=RuleOutcome.FAIL, fail_effect=Outcome.REFER_FOR_REVIEW, severity=Severity.WARNING, explanation=explanation)


def test_handbook_has_one_section_per_rule(handbook):
    rule_sections = [p.rule_id for p in handbook.passages if p.rule_id]
    assert sorted(rule_sections) == sorted(spec.rule_id for spec in RULES)
    assert handbook.policy_version == "demo-policy-1.0"


def test_handbook_values_come_from_policy_config(handbook, policy):
    assert not any(re.search(r"\{[a-z_]+\}", p.text) for p in handbook.passages)
    assert f"no more than {policy.thresholds.max_pay_stub_age_days} days" in handbook.get("R012_DOCUMENT_RECENCY").text
    changed = policy.model_copy(update={"thresholds": policy.thresholds.model_copy(update={"max_pay_stub_age_days": 30})})
    assert "no more than 30 days" in load_handbook(Settings().handbook_path, changed).get("R012_DOCUMENT_RECENCY").text


def test_handbook_rejects_other_policy_version_and_unknown_placeholders(policy):
    with pytest.raises(ValueError, match="targets policy"):
        parse_handbook("---\nhandbook_version: x\npolicy_version: other\n---\n## [GEN_X] X\ntext", policy)
    with pytest.raises(ValueError, match="placeholder"):
        parse_handbook(f"---\nhandbook_version: x\npolicy_version: {policy.policy_version}\n---\n## [GEN_X] X\n{{nope}}", policy)


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The deposits were dated") == ["deposit", "dat"]
    assert tokenize("pay_stub YTD 2026") == ["pay", "stub", "ytd"]


def test_bm25_ranks_matching_document_first():
    scorer = BM25Scorer(["bank deposits and net pay", "employment letter salary", "pay stub recency"])
    scores = scorer.scores("salary on the letter")
    assert scores.index(max(scores)) == 1
    assert scorer.scores("unrelated zebra") == [0.0, 0.0, 0.0]


def test_search_answers_reviewer_question(retriever):
    hits = retriever.search("How old can the most recent pay stub be?", k=3)
    assert hits[0].passage.passage_id == "R012_DOCUMENT_RECENCY" and hits[0].rank == 1
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_cite_and_annotate_only_failed_rules(retriever):
    stale = failed("R012_DOCUMENT_RECENCY", "Pay stub evidence is recent", "Most recent pay stub is 139 days old (limit 60); a recent pay stub is required")
    passed = stale.model_copy(update={"rule_id": "R001_REQUIRED_DOCUMENTS", "outcome": RuleOutcome.PASS})
    annotated = retriever.annotate([stale, passed])
    assert annotated[0].citation.passage_id == "R012_DOCUMENT_RECENCY"
    assert annotated[0].citation.matches_rule and annotated[0].citation.retriever == "demo-handbook-1.0/bm25"
    assert len(annotated[0].citation.excerpt) <= 283
    assert annotated[1].citation is None


def test_mismatched_citation_is_flagged(retriever):
    # A citation is retrieved from text, not looked up by ID, so a wrong passage is visible as matches_rule=False.
    odd = failed("R004_NO_DUPLICATE_DOCUMENTS", "Something", "Unsupported currency: USD (allowed: CAD)")
    assert retriever.cite(odd).passage_id == "R014_CURRENCY_SUPPORTED"
    assert retriever.cite(odd).matches_rule is False


class HashingEncoder:
    """Deterministic stand-in for a sentence-transformer: bag of hashed tokens."""

    name = "fake-hashing"

    def encode(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 64
            for tok in tokenize(text):
                vec[sum(map(ord, tok)) % 64] += 1.0
            out.append(vec)
        return out


def test_embedding_backend_with_injected_encoder(policy):
    settings = dataclasses.replace(Settings(), retrieval_backend="embedding")
    r = build_retriever(settings, policy, encoder=HashingEncoder())
    assert isinstance(r.scorer, EmbeddingScorer)
    assert r.version == "demo-handbook-1.0/embedding:fake-hashing"
    assert r.search("Bank statement deposits match net pay", k=1)[0].passage.passage_id == "R015_DEPOSITS_CORROBORATE"


def test_factory_none_and_unknown(policy):
    assert build_retriever(dataclasses.replace(Settings(), retrieval_backend="none"), policy) is None
    with pytest.raises(ValueError, match="unknown"):
        build_retriever(dataclasses.replace(Settings(), retrieval_backend="faiss"), policy)


def test_empty_scores_return_no_hits(handbook):
    class Zero:
        name = "zero"

        def scores(self, query):
            return [0.0] * len(handbook.passages)

    assert PolicyRetriever(handbook, Zero()).search("anything") == []


def test_retrieval_eval_meets_bm25_thresholds(tmp_path):
    report = retrieval_eval.run(dataclasses.replace(Settings(), retrieval_backend="bm25", reports_dir=tmp_path))
    assert report["thresholds"] and all(t["passed"] for t in report["thresholds"]), report["thresholds"]
    assert report["metrics"]["rule_citations"]["queries"] > 20
