"""Evaluation metrics. Every metric states its numerator and denominator.

Comparison rules (see EVALUATION.md):
* money: canonical Decimal equality to the cent
* dates: ISO date equality
* enums: exact canonical value
* names: case-insensitive after collapsing whitespace and trailing punctuation only
* null vs value is always a mismatch, in either direction
"""

from decimal import Decimal
from typing import Any

import pandas as pd

from app.extraction.normalize import to_iso
from app.models.common import CRITICAL_FIELDS_BY_TYPE, FIELDS_BY_TYPE, MONEY_FIELDS, SCALAR_FIELDS, TEXT_FIELDS, DocumentType
from app.models.extraction import DocumentExtraction

CRITICAL_WEIGHT = 2.0
DEFAULT_WEIGHT = 1.0


def _norm_text(value: str) -> str:
    return " ".join(value.split()).strip(" .,").casefold()


def values_match(field: str, expected: str | None, actual: str | None) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if field in MONEY_FIELDS:
        return Decimal(expected) == Decimal(actual)
    if field in TEXT_FIELDS:
        return _norm_text(expected) == _norm_text(actual)
    return expected == actual


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


# ---------------------------------------------------------------- extraction


def field_rows(case_id: str, document_path: str, expected: dict[str, Any], actual: DocumentExtraction) -> list[dict[str, Any]]:
    """One row per field applicable to the *expected* document type."""
    expected_type = DocumentType(expected["document_type"])
    critical = set(CRITICAL_FIELDS_BY_TYPE[expected_type])
    rows = []
    for field in FIELDS_BY_TYPE[expected_type]:
        if field == "deposits":
            exp = sorted((d["date"], str(Decimal(d["amount"]))) for d in expected.get("deposits", []))
            act = sorted((d.date.isoformat(), str(d.amount)) for d in actual.deposits)
            rows.append(
                {
                    "case_id": case_id, "document": document_path, "doc_type": expected_type.value, "field": field,
                    "expected": "; ".join(f"{d} {a}" for d, a in exp) or None, "actual": "; ".join(f"{d} {a}" for d, a in act) or None,
                    "match": exp == act, "critical": False, "money": False, "expected_null": not exp, "actual_null": not act,
                    "abs_error": None, "deposits_expected": len(exp), "deposits_actual": len(act), "deposits_correct": len(set(exp) & set(act)),
                }
            )
            continue
        exp_v = expected.get(field)
        act_v = to_iso(actual.value(field)) if field in SCALAR_FIELDS else None
        abs_error = None
        if field in MONEY_FIELDS and exp_v is not None and act_v is not None:
            abs_error = float(abs(Decimal(exp_v) - Decimal(act_v)))
        rows.append(
            {
                "case_id": case_id, "document": document_path, "doc_type": expected_type.value, "field": field,
                "expected": exp_v, "actual": act_v, "match": values_match(field, exp_v, act_v),
                "critical": field in critical, "money": field in MONEY_FIELDS, "expected_null": exp_v is None, "actual_null": act_v is None,
                "abs_error": abs_error, "deposits_expected": None, "deposits_actual": None, "deposits_correct": None,
            }
        )
    return rows


def classification_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """pairs: (expected_type, actual_type) per document."""
    labels = [t.value for t in DocumentType]
    per_class = {}
    for label in labels:
        tp = sum(1 for e, a in pairs if e == label and a == label)
        fp = sum(1 for e, a in pairs if e != label and a == label)
        fn = sum(1 for e, a in pairs if e == label and a != label)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else (0.0 if (tp + fp + fn) else None)
        per_class[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    scored = [v["f1"] for v in per_class.values() if v["support"] and v["f1"] is not None]
    return {
        "documents": len(pairs),
        "accuracy": _ratio(sum(1 for e, a in pairs if e == a), len(pairs)),
        "macro_f1": round(sum(scored) / len(scored), 4) if scored else None,
        "per_class": per_class,
        "definition": "Per-class precision = TP/(TP+FP), recall = TP/(TP+FN) over documents; macro F1 averages classes with support > 0.",
    }


def extraction_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"field_rows": 0}
    df = df.copy()
    df["weight"] = df["critical"].map({True: CRITICAL_WEIGHT, False: DEFAULT_WEIGHT})
    per_field = (
        df.groupby("field")
        .agg(rows=("match", "size"), correct=("match", "sum"))
        .assign(accuracy=lambda x: (x["correct"] / x["rows"]).round(4))
        .reset_index()
        .sort_values("field")
    )
    per_type = (
        df.groupby("doc_type")
        .agg(rows=("match", "size"), correct=("match", "sum"))
        .assign(accuracy=lambda x: (x["correct"] / x["rows"]).round(4))
        .reset_index()
    )
    money = df[df["money"] & ~df["expected_null"]]
    both = df[df["money"] & df["abs_error"].notna()]
    expected_null = df[df["expected_null"]]
    expected_present = df[~df["expected_null"]]
    critical = df[df["critical"]]
    deposits = df[df["field"] == "deposits"]
    dep_exp, dep_act, dep_ok = (int(deposits[c].sum()) if not deposits.empty else 0 for c in ("deposits_expected", "deposits_actual", "deposits_correct"))
    return {
        "field_rows": int(len(df)),
        "overall_field_accuracy": _ratio(int(df["match"].sum()), len(df)),
        "field_weighted_score": _ratio(float((df["weight"] * df["match"]).sum()), float(df["weight"].sum())),
        "critical_field_error_rate": _ratio(int((~critical["match"]).sum()), len(critical)),
        "missing_field_detection_rate": _ratio(int(expected_null["actual_null"].sum()), len(expected_null)),
        "unsupported_value_rate": _ratio(int((~expected_null["actual_null"]).sum()), len(expected_null)),
        "present_field_recall": _ratio(int((~expected_present["actual_null"]).sum()), len(expected_present)),
        "money_exact_match_rate": _ratio(int(money["match"].sum()), len(money)),
        "money_mean_abs_error": round(float(both["abs_error"].mean()), 2) if not both.empty else None,
        "deposit_precision": _ratio(dep_ok, dep_act),
        "deposit_recall": _ratio(dep_ok, dep_exp),
        "per_field": per_field.to_dict(orient="records"),
        "per_document_type": per_type.to_dict(orient="records"),
        "definitions": {
            "overall_field_accuracy": "matching (document, field) pairs / all pairs applicable to the expected document type",
            "field_weighted_score": f"weighted matches / total weight; critical fields weight {CRITICAL_WEIGHT}, others {DEFAULT_WEIGHT}",
            "critical_field_error_rate": "mismatched critical (document, field) pairs / critical pairs",
            "missing_field_detection_rate": "pairs where label is null and extractor returned null / pairs where label is null",
            "unsupported_value_rate": "pairs where label is null but extractor returned a value / pairs where label is null",
            "present_field_recall": "pairs where label has a value and extractor returned any value / pairs where label has a value",
            "money_exact_match_rate": "money pairs equal to the cent / money pairs with a labeled value",
            "money_mean_abs_error": "mean |expected - actual| over money pairs where both are non-null",
            "deposit_precision": "correct (date, amount) deposits / extracted deposits",
            "deposit_recall": "correct (date, amount) deposits / labeled deposits",
        },
    }


# -------------------------------------------------------------------- policy


def policy_metrics(cases: list[dict[str, Any]], rule_ids: list[str], insufficient_rules: set[str]) -> dict[str, Any]:
    """cases: dicts with expected_outcome, actual_outcome, expected_failed_rules, actual_failed_rules."""
    n = len(cases)
    exp_not_approve = [c for c in cases if c["expected_outcome"] != "APPROVE"]
    exp_not_refer = [c for c in cases if c["expected_outcome"] != "REFER_FOR_REVIEW"]
    needs_evidence = [c for c in cases if set(c["expected_failed_rules"]) & insufficient_rules]
    rule_pairs = [(rid in c["expected_failed_rules"], rid in c["actual_failed_rules"]) for c in cases for rid in rule_ids]
    outcomes = ["APPROVE", "REFER_FOR_REVIEW", "INSUFFICIENT_EVIDENCE"]
    confusion = {e: {a: sum(1 for c in cases if c["expected_outcome"] == e and c["actual_outcome"] == a) for a in outcomes} for e in outcomes}
    per_rule = {}
    for rid in rule_ids:
        pairs = [(rid in c["expected_failed_rules"], rid in c["actual_failed_rules"]) for c in cases]
        per_rule[rid] = {
            "expected_fail": sum(1 for e, _ in pairs if e),
            "actual_fail": sum(1 for _, a in pairs if a),
            "agreement": _ratio(sum(1 for e, a in pairs if e == a), len(pairs)),
        }
    return {
        "cases": n,
        "outcome_agreement": _ratio(sum(1 for c in cases if c["expected_outcome"] == c["actual_outcome"]), n),
        "rule_level_agreement": _ratio(sum(1 for e, a in rule_pairs if e == a), len(rule_pairs)),
        "false_approval_rate": _ratio(sum(1 for c in exp_not_approve if c["actual_outcome"] == "APPROVE"), len(exp_not_approve)),
        "incorrect_escalation_rate": _ratio(sum(1 for c in exp_not_refer if c["actual_outcome"] == "REFER_FOR_REVIEW"), len(exp_not_refer)),
        "missing_evidence_detection_rate": _ratio(sum(1 for c in needs_evidence if set(c["actual_failed_rules"]) & insufficient_rules), len(needs_evidence)),
        "confusion_matrix": confusion,
        "per_rule": per_rule,
        "definitions": {
            "outcome_agreement": "cases whose final outcome equals the label / all cases",
            "rule_level_agreement": "(case, rule) pairs where FAIL vs not-FAIL equals the label / cases x rules",
            "false_approval_rate": "cases output APPROVE whose label is not APPROVE / cases whose label is not APPROVE",
            "incorrect_escalation_rate": "cases output REFER_FOR_REVIEW whose label is not REFER / cases whose label is not REFER",
            "missing_evidence_detection_rate": f"cases labeled with a failing evidence rule ({', '.join(sorted(insufficient_rules))}) where at least one such rule failed / those cases",
        },
    }


def derived_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    income_ok = sum(1 for c in cases if c["expected_income"] == c["actual_income"] or (c["expected_income"] and c["actual_income"] and Decimal(c["expected_income"]) == Decimal(c["actual_income"])))
    return {
        "cases": n,
        "recurring_income_agreement": _ratio(income_ok, n),
        "income_pattern_agreement": _ratio(sum(1 for c in cases if c["expected_pattern"] == c["actual_pattern"]), n),
        "definitions": {
            "recurring_income_agreement": "cases whose recurring annual income equals the label to the cent (both null counts as agreement) / all cases",
            "income_pattern_agreement": "cases whose RECURRING/VARIABLE/UNKNOWN pattern equals the label / all cases",
        },
    }
