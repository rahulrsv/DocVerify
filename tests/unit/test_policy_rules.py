"""One test per rule: each rule fails on its trigger and passes (or is N/A) otherwise."""

import dataclasses
from datetime import date
from decimal import Decimal

import pytest

from app.calculations.income import calculate_case
from app.models.case import DocumentRecord
from app.models.common import DocumentType, EmploymentStatus, Outcome, PayFrequency, RuleOutcome
from app.models.extraction import BankDeposit
from app.policies.config import PolicyConfig, RuleOverride
from app.policies.engine import decide, evaluate_rules
from app.policies.rules import RULES, PolicyContext
from tests.conftest import AS_OF, letter, make_extraction, record, stub


def run(extractions, policy, records=None):
    records = records or [record(e.document_id) for e in extractions]
    calc = calculate_case(records, extractions, AS_OF, policy.calculation_settings())
    results = evaluate_rules(PolicyContext(records, extractions, calc, AS_OF), policy)
    return {r.rule_id: r for r in results}, decide(results, policy)


def failed(results):
    return {rid for rid, r in results.items() if r.outcome is RuleOutcome.FAIL}


def test_every_rule_has_unique_id_and_version():
    ids = [r.rule_id for r in RULES]
    assert len(ids) == len(set(ids)) == 15
    assert all(r.version for r in RULES)


def test_clean_case_approves(policy):
    results, decision = run([stub(), letter()], policy)
    assert failed(results) == set()
    assert decision.outcome is Outcome.APPROVE
    assert "not credit decisions" in decision.disclaimer
    assert all(r.explanation for r in results.values())


BANK_OK = [BankDeposit(date=date(2026, 10, 2), amount=Decimal("2284.11"), evidence="x", source_document_id="D3")]


@pytest.mark.parametrize(
    ("rule_id", "extractions", "records", "outcome"),
    [
        ("R001_REQUIRED_DOCUMENTS", [stub()], None, Outcome.INSUFFICIENT_EVIDENCE),
        ("R002_CRITICAL_FIELDS_PRESENT", [stub(pay_date=None, missing=["pay_date"]), letter()], None, Outcome.INSUFFICIENT_EVIDENCE),
        ("R003_DOCUMENTS_READABLE", [stub(), letter(), make_extraction("D3", DocumentType.UNKNOWN)], None, Outcome.REFER_FOR_REVIEW),
        ("R004_NO_DUPLICATE_DOCUMENTS", [stub("D1"), stub("D2"), letter("D3")], None, Outcome.REFER_FOR_REVIEW),
        ("R005_DATES_AND_AMOUNTS_PLAUSIBLE", [stub(start=date(2026, 9, 27), end=date(2026, 9, 14)), letter()], None, Outcome.REFER_FOR_REVIEW),
        ("R006_IDENTITY_CONSISTENT", [stub(), letter(employee_name="Someone Else")], None, Outcome.REFER_FOR_REVIEW),
        ("R007_EMPLOYER_CONSISTENT", [stub(), letter(employer_name="Other Corp.")], None, Outcome.REFER_FOR_REVIEW),
        ("R008_SALARY_CONSISTENT", [stub(), letter(salary="120000.00")], None, Outcome.REFER_FOR_REVIEW),
        ("R009_INCOME_RECURRING", [stub(freq=PayFrequency.IRREGULAR), letter()], None, Outcome.REFER_FOR_REVIEW),
        ("R010_INCOME_THRESHOLD", [stub(gross="1000.00", net_pay=Decimal("800.00")), letter(salary=None)], None, Outcome.REFER_FOR_REVIEW),
        ("R011_EMPLOYMENT_ACTIVE", [stub(), letter(employment_status=EmploymentStatus.TERMINATED)], None, Outcome.REFER_FOR_REVIEW),
        ("R012_DOCUMENT_RECENCY", [stub(start=date(2026, 5, 1), end=date(2026, 5, 14), paid=date(2026, 5, 19)), letter()], None, Outcome.INSUFFICIENT_EVIDENCE),
        ("R013_NO_MANIPULATION_SIGNALS", [stub(), letter()], [record("D1", injection_signals=["SYSTEM_OVERRIDE"]), record("D2")], Outcome.REFER_FOR_REVIEW),
        ("R014_CURRENCY_SUPPORTED", [stub(currency="USD"), letter(currency="USD")], None, Outcome.REFER_FOR_REVIEW),
        (
            "R015_DEPOSITS_CORROBORATE",
            [stub(), letter(), make_extraction("D3", DocumentType.BANK_STATEMENT, employee_name="Jordan Avery", currency="CAD",
                                               deposits=[BankDeposit(date=date(2026, 10, 2), amount=Decimal("1.00"), evidence="x", source_document_id="D3")])],
            None,
            Outcome.REFER_FOR_REVIEW,
        ),
    ],
)
def test_rule_fails_on_trigger(policy, rule_id, extractions, records, outcome):
    results, decision = run(extractions, policy, records)
    assert results[rule_id].outcome is RuleOutcome.FAIL, results[rule_id].explanation
    assert results[rule_id].fail_effect is outcome
    assert results[rule_id].explanation
    assert decision.outcome is not Outcome.APPROVE


def test_rule_passes_with_corroborating_bank(policy):
    bank = make_extraction("D3", DocumentType.BANK_STATEMENT, employee_name="Jordan Avery", currency="CAD", deposits=BANK_OK)
    results, decision = run([stub(), letter(), bank], policy)
    assert results["R015_DEPOSITS_CORROBORATE"].outcome is RuleOutcome.PASS
    assert decision.outcome is Outcome.APPROVE


def test_not_applicable_rules_on_letter_only(policy):
    results, _ = run([letter()], policy)
    for rid in ("R008_SALARY_CONSISTENT", "R009_INCOME_RECURRING", "R010_INCOME_THRESHOLD", "R012_DOCUMENT_RECENCY", "R015_DEPOSITS_CORROBORATE"):
        assert results[rid].outcome is RuleOutcome.NOT_APPLICABLE


def test_model_self_report_can_only_escalate(policy):
    s = stub().model_copy(update={"model_flagged_suspicious": True})
    results, decision = run([s, letter()], policy)
    assert results["R013_NO_MANIPULATION_SIGNALS"].outcome is RuleOutcome.FAIL
    assert decision.outcome is Outcome.REFER_FOR_REVIEW


def test_precedence_refer_over_insufficient(policy):
    # missing letter (INSUFFICIENT) + unreadable doc (REFER) -> REFER by configured precedence
    recs = [record("D1"), DocumentRecord(document_id="D2", filename="x.pdf", media_type="application/pdf", sha256="x", load_error="NO_TEXT_LAYER")]
    results, decision = run([stub(), make_extraction("D2", DocumentType.UNKNOWN)], policy, recs)
    assert {"R001_REQUIRED_DOCUMENTS", "R003_DOCUMENTS_READABLE"} <= failed(results)
    assert decision.outcome is Outcome.REFER_FOR_REVIEW
    assert decision.reasons[0].startswith("[R003")


def test_threshold_change_via_config_only(policy):
    raised = policy.model_copy(update={"thresholds": policy.thresholds.model_copy(update={"min_recurring_annual_income": Decimal("90000")})})
    _, base = run([stub(), letter()], policy)
    results, decision = run([stub(), letter()], raised)
    assert base.outcome is Outcome.APPROVE
    assert results["R010_INCOME_THRESHOLD"].outcome is RuleOutcome.FAIL
    assert decision.outcome is Outcome.REFER_FOR_REVIEW


def test_disable_and_override_rules(policy):
    cfg = policy.model_copy(update={"disabled_rules": ["R001_REQUIRED_DOCUMENTS"], "rule_overrides": {"R010_INCOME_THRESHOLD": RuleOverride(fail_effect=Outcome.INSUFFICIENT_EVIDENCE)}})
    results, decision = run([stub(gross="1000.00", net_pay=Decimal("800.00"))], cfg)
    assert "R001_REQUIRED_DOCUMENTS" not in results
    assert decision.outcome is Outcome.INSUFFICIENT_EVIDENCE


def test_invalid_precedence_rejected(policy):
    data = policy.model_dump()
    data["outcome_precedence"] = ["APPROVE", "APPROVE", "REFER_FOR_REVIEW"]
    with pytest.raises(ValueError):
        PolicyConfig.model_validate(data)


def test_policy_is_deterministic(policy):
    a = run([stub(), letter(salary="120000.00")], policy)
    b = run([stub(), letter(salary="120000.00")], policy)
    assert [r.model_dump() for r in a[0].values()] == [r.model_dump() for r in b[0].values()]
    assert a[1] == b[1]
    assert dataclasses.is_dataclass(PolicyContext)
