from decimal import Decimal

from app.core.pipeline import CaseProcessor, DocumentInput
from app.extraction.base import ProviderError, ProviderResult
from app.models.common import Outcome, RuleOutcome
from tests.conftest import AS_OF, doc_bytes


def inputs(case_id, *names):
    return [DocumentInput(n, doc_bytes(case_id, n)) for n in names]


def test_end_to_end_clean_case(processor, audit):
    result = processor.process_case("GC001", inputs("GC001", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert result.decision.outcome is Outcome.APPROVE
    assert result.calculations.recurring_annual_income.value == Decimal("79999.92")
    stub = result.extractions[0]
    assert stub.gross_pay.evidence == "Gross Pay: $3,076.92"
    assert stub.gross_pay.source_document_id == result.documents[0].document_id
    assert result.metadata.provider == "mock" and result.metadata.policy_version == "demo-policy-1.0"
    assert len(result.rule_results) == 15

    stored = audit.latest_run("GC001")
    assert stored.run_id == result.run_id
    types = [e["event_type"] for e in audit.events_for_case("GC001")]
    assert types[0] == "CASE_RECEIVED" and types[-1] == "DECISION_RECORDED"
    assert {"DOCUMENT_LOADED", "EXTRACTION_COMPLETED", "CALCULATIONS_COMPLETED", "POLICY_EVALUATED"} <= set(types)
    assert audit.verify_chain()[0]


def test_audit_payloads_do_not_store_document_text(processor, audit):
    processor.process_case("GC001", inputs("GC001", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    blob = str(audit.events_for_case("GC001"))
    assert "Gross Pay: $3,076.92" not in blob


def test_idempotent_replay_and_force(processor, audit):
    docs = inputs("GC001", "pay_stub.txt", "employment_letter.txt")
    first = processor.process_case("C1", docs, as_of_date=AS_OF)
    replay = processor.process_case("C1", docs, as_of_date=AS_OF)
    assert replay.idempotent_replay and replay.run_id == first.run_id
    forced = processor.process_case("C1", docs, as_of_date=AS_OF, force=True)
    assert forced.run_id != first.run_id
    assert len(audit.runs_for_case("C1")) == 2
    assert "PROCESSING_REPLAYED" in [e["event_type"] for e in audit.events_for_case("C1")]


def test_prompt_injection_case_is_referred_and_value_preserved(processor):
    result = processor.process_case("GC044", inputs("GC044", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert result.decision.outcome is Outcome.REFER_FOR_REVIEW
    assert result.extractions[0].gross_pay.value == Decimal("1923.08")
    r013 = next(r for r in result.rule_results if r.rule_id == "R013_NO_MANIPULATION_SIGNALS")
    assert r013.outcome is RuleOutcome.FAIL
    assert result.documents[0].injection_signals


def test_duplicate_upload_detected(processor):
    docs = inputs("GC001", "pay_stub.txt", "pay_stub.txt", "employment_letter.txt")
    result = processor.process_case("DUP", docs, as_of_date=AS_OF)
    assert result.decision.outcome is Outcome.REFER_FOR_REVIEW
    assert result.calculations.has("DUPLICATE_DOCUMENT")


def test_conflicting_and_missing_evidence(processor):
    conflict = processor.process_case("GC019", inputs("GC019", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert conflict.decision.outcome is Outcome.REFER_FOR_REVIEW
    missing = processor.process_case("GC013", inputs("GC013", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert missing.decision.outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert missing.extractions[0].gross_pay.value is None


class FailingProvider:
    name, model, prompt_version = "failing", "none", "v0"

    def extract(self, text, document_id):
        raise ProviderError("timeout", "simulated timeout", attempts=3)


class SuggestApproveProvider:
    """A compromised model that returns fabricated, ungrounded values."""

    name, model, prompt_version = "evil", "none", "v0"

    def extract(self, text, document_id):
        from app.extraction.mock_provider import RuleBasedExtractionProvider
        from app.models.extraction import RawField

        raw = RuleBasedExtractionProvider().extract(text, document_id).raw
        if raw.document_type == "pay_stub":
            raw.gross_pay = RawField(value="15000.00", evidence="Gross Pay: $15,000.00", confidence=1.0)
        return ProviderResult(raw=raw)


def test_provider_failure_degrades_to_insufficient_evidence(policy, audit):
    proc = CaseProcessor(FailingProvider(), policy, audit)
    result = proc.process_case("F1", inputs("GC001", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert result.decision.outcome is not Outcome.APPROVE
    assert result.metadata.extraction_attempts[result.documents[0].document_id] == 3
    assert any("timeout" in e for e in result.metadata.errors)
    assert "EXTRACTION_FAILED" in [e["event_type"] for e in audit.events_for_case("F1")]


def test_fabricated_model_values_cannot_force_approval(policy, audit):
    proc = CaseProcessor(SuggestApproveProvider(), policy, audit)
    result = proc.process_case("E1", inputs("GC047", "pay_stub.txt", "employment_letter.txt"), as_of_date=AS_OF)
    assert result.extractions[0].gross_pay.value is None  # ungrounded value discarded
    assert result.decision.outcome is Outcome.INSUFFICIENT_EVIDENCE


def test_pdf_case(processor):
    result = processor.process_case("GC006", inputs("GC006", "pay_stub_1.pdf", "pay_stub_2.pdf", "employment_letter.pdf"), as_of_date=AS_OF)
    assert result.decision.outcome is Outcome.APPROVE
    assert result.calculations.gross_pay_variation_pct.value == Decimal("0.00")


def test_failed_rules_cite_handbook_without_changing_the_decision(audit, policy, tmp_path):
    import dataclasses

    from app.audit.store import AuditStore
    from app.core.config import Settings
    from app.extraction.mock_provider import RuleBasedExtractionProvider
    from app.retrieval.factory import build_retriever

    retriever = build_retriever(dataclasses.replace(Settings(), retrieval_backend="bm25"), policy)
    docs = inputs("GC016", "pay_stub.txt", "employment_letter.txt")
    with_cites = CaseProcessor(RuleBasedExtractionProvider(), policy, audit, retriever=retriever).process_case("GC016", docs, as_of_date=AS_OF)
    plain = CaseProcessor(RuleBasedExtractionProvider(), policy, AuditStore(tmp_path / "plain.db")).process_case("GC016", docs, as_of_date=AS_OF)

    assert with_cites.decision == plain.decision
    assert [(r.rule_id, r.outcome) for r in with_cites.rule_results] == [(r.rule_id, r.outcome) for r in plain.rule_results]
    stale = next(r for r in with_cites.rule_results if r.rule_id == "R012_DOCUMENT_RECENCY")
    assert stale.outcome is RuleOutcome.FAIL and stale.citation.passage_id == "R012_DOCUMENT_RECENCY"
    assert all(r.citation is None for r in with_cites.rule_results if r.outcome is not RuleOutcome.FAIL)
    assert with_cites.metadata.retrieval_version == "demo-handbook-1.0/bm25" and plain.metadata.retrieval_version is None
    assert with_cites.input_fingerprint != plain.input_fingerprint
    event = next(e for e in audit.events_for_case("GC016") if e["event_type"] == "POLICY_CITATIONS_ATTACHED")
    assert event["payload"]["citations"] == {"R012_DOCUMENT_RECENCY": "R012_DOCUMENT_RECENCY"}
