import copy

import pandas as pd

from app.evals.compare import check_thresholds, compare_to_baseline
from app.evals.metrics import classification_metrics, extraction_metrics, field_rows, policy_metrics, values_match
from app.models.common import DocumentType
from tests.conftest import stub


def test_values_match_normalization_is_narrow():
    assert values_match("gross_pay", "3076.90", "3076.9")
    assert not values_match("gross_pay", "3076.92", "3076.93")
    assert values_match("employer_name", "Northwind Logistics Inc.", "northwind  logistics inc")
    assert not values_match("employer_name", "Northwind Logistics Inc.", "Northwind Logistic Inc.")
    assert not values_match("pay_date", "2026-10-02", None)
    assert values_match("pay_date", None, None)


def test_field_rows_scores_against_expected_type():
    expected = {"document_type": "pay_stub", "employee_name": "Jordan Avery", "employer_name": "Northwind Logistics Inc.", "pay_period_start": "2026-09-14", "pay_period_end": "2026-09-27",
                "pay_date": "2026-10-02", "gross_pay": "3076.92", "net_pay": "2284.11", "pay_frequency": "BIWEEKLY", "ytd_gross": "61538.40", "currency": "CAD"}
    rows = field_rows("C", "C/d", expected, stub())
    by_field = {r["field"]: r for r in rows}
    assert len(rows) == 10
    assert by_field["gross_pay"]["match"]
    assert not by_field["ytd_gross"]["match"]  # stub() uses 60000.00
    assert by_field["ytd_gross"]["abs_error"] == 1538.40


def test_extraction_metrics_definitions():
    df = pd.DataFrame(
        [
            {"field": "gross_pay", "doc_type": "pay_stub", "match": True, "critical": True, "money": True, "expected_null": False, "actual_null": False, "abs_error": 0.0, "deposits_expected": None, "deposits_actual": None, "deposits_correct": None},
            {"field": "net_pay", "doc_type": "pay_stub", "match": False, "critical": False, "money": True, "expected_null": True, "actual_null": False, "abs_error": None, "deposits_expected": None, "deposits_actual": None, "deposits_correct": None},
            {"field": "pay_date", "doc_type": "pay_stub", "match": False, "critical": True, "money": False, "expected_null": False, "actual_null": True, "abs_error": None, "deposits_expected": None, "deposits_actual": None, "deposits_correct": None},
        ]
    )
    m = extraction_metrics(df)
    assert m["overall_field_accuracy"] == round(1 / 3, 4)
    assert m["field_weighted_score"] == round(2 / 5, 4)
    assert m["critical_field_error_rate"] == 0.5
    assert m["missing_field_detection_rate"] == 0.0
    assert m["unsupported_value_rate"] == 1.0
    assert m["present_field_recall"] == 0.5


def test_classification_metrics():
    pairs = [("pay_stub", "pay_stub"), ("pay_stub", "unknown"), ("employment_letter", "employment_letter")]
    m = classification_metrics(pairs)
    assert m["per_class"]["pay_stub"]["recall"] == 0.5
    assert m["per_class"]["pay_stub"]["precision"] == 1.0
    assert m["per_class"][DocumentType.UNKNOWN.value]["precision"] == 0.0
    assert m["accuracy"] == round(2 / 3, 4)


def test_policy_metrics():
    cases = [
        {"expected_outcome": "REFER_FOR_REVIEW", "actual_outcome": "APPROVE", "expected_failed_rules": ["R008"], "actual_failed_rules": []},
        {"expected_outcome": "APPROVE", "actual_outcome": "APPROVE", "expected_failed_rules": [], "actual_failed_rules": []},
        {"expected_outcome": "INSUFFICIENT_EVIDENCE", "actual_outcome": "REFER_FOR_REVIEW", "expected_failed_rules": ["R001"], "actual_failed_rules": ["R001", "R003"]},
    ]
    m = policy_metrics(cases, ["R001", "R003", "R008"], {"R001"})
    assert m["outcome_agreement"] == round(1 / 3, 4)
    assert m["false_approval_rate"] == 0.5
    assert m["incorrect_escalation_rate"] == 0.5
    assert m["missing_evidence_detection_rate"] == 1.0
    assert m["rule_level_agreement"] == round(7 / 9, 4)


def test_threshold_failure_detected():
    results = check_thresholds({"policy_end_to_end": {"false_approval_rate": 0.02}}, {"policy_end_to_end.false_approval_rate": {"max": 0.0}, "missing.metric": {"min": 1}})
    assert [r["passed"] for r in results] == [False, False]


def _report(accuracy=1.0, passed=True, outcome="APPROVE"):
    return {
        "metadata": {"eval_run_id": "e", "provider": "mock", "dataset_sha256": "x"},
        "metrics": {"extraction": {"overall_field_accuracy": accuracy, "per_field": [{"field": "gross_pay", "accuracy": accuracy}]}, "policy_end_to_end": {"false_approval_rate": 0.0}},
        "cases": [{"case_id": "GC001", "passed": passed, "expected_outcome": "APPROVE", "actual_outcome": outcome, "rule_diff": ""}],
    }


def test_baseline_regressions_detected():
    base = _report()
    current = _report(accuracy=0.9, passed=False, outcome="REFER_FOR_REVIEW")
    cmp = compare_to_baseline(current, base)
    assert cmp["has_regressions"]
    assert cmp["newly_failing_cases"][0]["case_id"] == "GC001"
    assert cmp["field_regressions"][0]["field"] == "gross_pay"
    assert any(d["regressed"] for d in cmp["metric_deltas"])
    assert not compare_to_baseline(copy.deepcopy(base), base)["has_regressions"]
    assert not compare_to_baseline(base, None)["baseline_found"]
