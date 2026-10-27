"""Baseline comparison and threshold checks."""

from typing import Any

# (metric path, higher_is_better)
HEADLINE_METRICS: list[tuple[str, bool]] = [
    ("extraction.overall_field_accuracy", True),
    ("extraction.field_weighted_score", True),
    ("extraction.critical_field_error_rate", False),
    ("extraction.missing_field_detection_rate", True),
    ("extraction.money_exact_match_rate", True),
    ("classification.macro_f1", True),
    ("derived.recurring_income_agreement", True),
    ("policy_end_to_end.outcome_agreement", True),
    ("policy_end_to_end.rule_level_agreement", True),
    ("policy_end_to_end.false_approval_rate", False),
    ("policy_end_to_end.incorrect_escalation_rate", False),
    ("policy_end_to_end.missing_evidence_detection_rate", True),
    ("policy_gold_extraction.outcome_agreement", True),
    ("policy_gold_extraction.rule_level_agreement", True),
]
METRIC_TOLERANCE = 0.005
FIELD_TOLERANCE = 0.01
_META_KEYS = ("provider", "model", "prompt_version", "schema_version", "policy_version", "dataset_version", "dataset_sha256")


def get_path(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def check_thresholds(metrics: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    results = []
    for path, bounds in thresholds.items():
        value = get_path(metrics, path)
        passed = value is not None
        if value is not None and "min" in bounds:
            passed = passed and value >= bounds["min"]
        if value is not None and "max" in bounds:
            passed = passed and value <= bounds["max"]
        results.append({"metric": path, "value": value, **bounds, "passed": passed})
    return results


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {"baseline_found": False, "has_regressions": False, "metric_deltas": [], "field_regressions": [], "newly_failing_cases": [], "newly_passing_cases": [], "metadata_changes": {}}

    meta_changes = {
        k: {"baseline": baseline["metadata"].get(k), "current": current["metadata"].get(k)}
        for k in _META_KEYS
        if baseline["metadata"].get(k) != current["metadata"].get(k)
    }
    deltas = []
    for path, higher_better in HEADLINE_METRICS:
        b, c = get_path(baseline["metrics"], path), get_path(current["metrics"], path)
        if b is None or c is None:
            continue
        delta = round(c - b, 4)
        regressed = (delta < -METRIC_TOLERANCE) if higher_better else (delta > METRIC_TOLERANCE)
        deltas.append({"metric": path, "baseline": b, "current": c, "delta": delta, "regressed": regressed})

    base_fields = {r["field"]: r["accuracy"] for r in baseline["metrics"]["extraction"].get("per_field", [])}
    field_regressions = [
        {"field": r["field"], "baseline": base_fields[r["field"]], "current": r["accuracy"], "delta": round(r["accuracy"] - base_fields[r["field"]], 4)}
        for r in current["metrics"]["extraction"].get("per_field", [])
        if r["field"] in base_fields and r["accuracy"] < base_fields[r["field"]] - FIELD_TOLERANCE
    ]

    base_cases = {c["case_id"]: c for c in baseline["cases"]}
    newly_failing, newly_passing = [], []
    for case in current["cases"]:
        prev = base_cases.get(case["case_id"])
        if prev is None:
            continue
        if prev["passed"] and not case["passed"]:
            newly_failing.append({"case_id": case["case_id"], "expected": case["expected_outcome"], "baseline_actual": prev["actual_outcome"], "current_actual": case["actual_outcome"], "rule_diff": case["rule_diff"]})
        elif not prev["passed"] and case["passed"]:
            newly_passing.append(case["case_id"])

    return {
        "baseline_found": True,
        "baseline_eval_run_id": baseline["metadata"].get("eval_run_id"),
        "baseline_created_at": baseline["metadata"].get("created_at"),
        "metadata_changes": meta_changes,
        "comparable": "provider" not in meta_changes and "dataset_sha256" not in meta_changes,
        "metric_deltas": deltas,
        "field_regressions": field_regressions,
        "newly_failing_cases": newly_failing,
        "newly_passing_cases": newly_passing,
        "has_regressions": bool(newly_failing or field_regressions or any(d["regressed"] for d in deltas)),
    }
