"""Evaluation orchestration shared by the CLI and the API."""

import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.audit.store import AuditStore
from app.calculations.income import calculate_case
from app.core.config import PROJECT_ROOT, Settings
from app.core.pipeline import CaseProcessor, DocumentInput
from app.documents.loader import load_document
from app.documents.safety import scan_for_injection
from app.evals.compare import check_thresholds, compare_to_baseline
from app.evals.dataset import GoldenCase, gold_extraction, load_dataset
from app.evals.metrics import classification_metrics, derived_metrics, extraction_metrics, field_rows, policy_metrics
from app.evals.report import render_markdown
from app.extraction.base import ExtractionProvider, ProviderError
from app.extraction.factory import build_provider
from app.extraction.normalize import to_iso
from app.models.case import RuleResult
from app.models.common import SCHEMA_VERSION, Outcome, RuleOutcome
from app.policies.config import PolicyConfig, load_policy
from app.policies.engine import decide, evaluate_rules
from app.policies.rules import RULES, PolicyContext

THRESHOLDS_FILE = "eval_thresholds.json"


class EvaluationError(Exception):
    pass


def _mode_label(provider: ExtractionProvider) -> str:
    if provider.name == "mock":
        return (
            f"MOCK MODE: metrics measure the deterministic rule-based extractor ({provider.model}), not an LLM. "
            "They say nothing about LLM extraction quality."
        )
    return f"LIVE LLM MODE: metrics measure {provider.name}/{provider.model} with prompt {provider.prompt_version} on synthetic data only."


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def gold_rule_results(case: GoldenCase, policy: PolicyConfig) -> list[RuleResult]:
    """Run calculations + policy on labeled extractions to isolate rule logic from extraction error."""
    records, extractions = [], []
    for index, doc in enumerate(case.documents):
        loaded = load_document(index, doc.path.name, doc.path.read_bytes())
        if loaded.text:
            loaded.record.injection_signals = scan_for_injection(loaded.text)
        records.append(loaded.record)
        extractions.append(gold_extraction(loaded.record.document_id, doc.expected))
    calcs = calculate_case(records, extractions, case.as_of_date, policy.calculation_settings())
    return evaluate_rules(PolicyContext(records, extractions, calcs, case.as_of_date), policy)


def _gold_policy_case(case: GoldenCase, policy: PolicyConfig) -> tuple[str, list[str]]:
    results = gold_rule_results(case, policy)
    return decide(results, policy).outcome.value, sorted(r.rule_id for r in results if r.outcome is RuleOutcome.FAIL)


def evaluate(
    processor: CaseProcessor,
    dataset_dir: Path,
    thresholds: dict[str, dict[str, float]],
    baseline: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = load_dataset(dataset_dir)
    cases = [c for c in dataset.cases if not case_ids or c.case_id in case_ids]
    if not cases:
        raise EvaluationError("No golden cases selected")
    eval_run_id = f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    rows: list[dict[str, Any]] = []
    class_pairs: list[tuple[str, str]] = []
    e2e_cases, gold_cases, derived_cases, case_results = [], [], [], []
    provider_errors = 0

    for case in cases:
        docs = [DocumentInput(d.path.name, d.path.read_bytes()) for d in case.documents]
        result = processor.process_case(case.case_id, docs, as_of_date=case.as_of_date, force=True, source="evaluation", evaluation_run_id=eval_run_id)
        provider_errors += sum(1 for e in result.extractions for i in e.issues if i.code.startswith("EXTRACTION_"))

        case_rows = []
        for doc, ext in zip(case.documents, result.extractions, strict=True):
            class_pairs.append((doc.expected["document_type"], ext.document_type.value))
            case_rows += field_rows(case.case_id, f"{case.case_id}/{doc.path.name}", doc.expected, ext)
        rows += case_rows

        actual_failed = sorted(r.rule_id for r in result.rule_results if r.outcome is RuleOutcome.FAIL)
        expected_failed = sorted(case.expected_failed_rules)
        gold_outcome, gold_failed = _gold_policy_case(case, processor.policy)
        e2e_cases.append({"expected_outcome": case.expected_outcome, "actual_outcome": result.decision.outcome.value, "expected_failed_rules": expected_failed, "actual_failed_rules": actual_failed})
        gold_cases.append({"expected_outcome": case.expected_outcome, "actual_outcome": gold_outcome, "expected_failed_rules": expected_failed, "actual_failed_rules": gold_failed})
        derived_cases.append(
            {
                "expected_income": case.expected_derived.get("recurring_annual_income"),
                "actual_income": to_iso(result.calculations.recurring_annual_income.value),
                "expected_pattern": case.expected_derived.get("income_pattern"),
                "actual_pattern": result.calculations.income_pattern,
            }
        )
        rule_diff = sorted([f"+{r}" for r in set(actual_failed) - set(expected_failed)] + [f"-{r}" for r in set(expected_failed) - set(actual_failed)])
        outcome_match = case.expected_outcome == result.decision.outcome.value
        case_results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "reason": case.reason,
                "run_id": result.run_id,
                "expected_outcome": case.expected_outcome,
                "actual_outcome": result.decision.outcome.value,
                "gold_extraction_outcome": gold_outcome,
                "expected_failed_rules": expected_failed,
                "actual_failed_rules": actual_failed,
                "rule_diff": " ".join(rule_diff),
                "outcome_match": outcome_match,
                "passed": outcome_match and not rule_diff,
                "field_accuracy": round(sum(r["match"] for r in case_rows) / len(case_rows), 4) if case_rows else None,
                "field_errors": [
                    {"document": r["document"], "field": r["field"], "expected": r["expected"], "actual": r["actual"], "critical": r["critical"]}
                    for r in case_rows
                    if not r["match"]
                ],
                "expected_income": case.expected_derived.get("recurring_annual_income"),
                "actual_income": to_iso(result.calculations.recurring_annual_income.value),
            }
        )

    rule_ids = [r.rule_id for r in RULES if r.rule_id not in processor.policy.disabled_rules]
    insufficient = {r.rule_id for r in RULES if r.fail_effect is Outcome.INSUFFICIENT_EVIDENCE}
    metrics = {
        "extraction": extraction_metrics(pd.DataFrame(rows)),
        "classification": classification_metrics(class_pairs),
        "derived": derived_metrics(derived_cases),
        "policy_end_to_end": policy_metrics(e2e_cases, rule_ids, insufficient),
        "policy_gold_extraction": policy_metrics(gold_cases, rule_ids, insufficient),
    }
    provider = processor.provider
    report: dict[str, Any] = {
        "metadata": {
            "eval_run_id": eval_run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": "mock" if provider.name == "mock" else "llm",
            "mode_label": _mode_label(provider),
            "provider": provider.name,
            "model": provider.model,
            "prompt_version": provider.prompt_version,
            "schema_version": SCHEMA_VERSION,
            "policy_version": processor.policy.policy_version,
            "dataset_version": dataset.version,
            "dataset_sha256": dataset.sha256,
            "dataset_integrity_ok": dataset.integrity_ok,
            "case_count": len(cases),
            "subset": bool(case_ids),
            "provider_errors": provider_errors,
            "git_commit": _git_commit(),
            "duration_s": round(time.perf_counter() - started, 2),
        },
        "metrics": metrics,
        "cases": case_results,
    }
    report["thresholds"] = check_thresholds(metrics, thresholds)
    report["comparison"] = compare_to_baseline(report, baseline)
    report["summary"] = {
        "eval_run_id": eval_run_id,
        "mode_label": report["metadata"]["mode_label"],
        "provider": provider.name,
        "model": provider.model,
        "cases": len(cases),
        "cases_passed": sum(1 for c in case_results if c["passed"]),
        "overall_field_accuracy": metrics["extraction"].get("overall_field_accuracy"),
        "classification_macro_f1": metrics["classification"]["macro_f1"],
        "e2e_outcome_agreement": metrics["policy_end_to_end"]["outcome_agreement"],
        "e2e_false_approval_rate": metrics["policy_end_to_end"]["false_approval_rate"],
        "gold_outcome_agreement": metrics["policy_gold_extraction"]["outcome_agreement"],
        "thresholds_passed": all(t["passed"] for t in report["thresholds"]),
        "failed_thresholds": [t["metric"] for t in report["thresholds"] if not t["passed"]],
        "has_regressions": report["comparison"]["has_regressions"],
        "newly_failing_cases": [c["case_id"] for c in report["comparison"]["newly_failing_cases"]],
    }
    return report


def load_thresholds(dataset_dir: Path) -> dict[str, dict[str, float]]:
    path = dataset_dir / THRESHOLDS_FILE
    return json.loads(path.read_text())["thresholds"] if path.exists() else {}


def baseline_path(reports_dir: Path, provider_name: str) -> Path:
    return reports_dir / "baselines" / f"{provider_name}.json"


def write_report(report: dict[str, Any], reports_dir: Path, update_baseline: bool = False) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, default=str)
    stamp = report["metadata"]["eval_run_id"]
    paths = {
        "report": reports_dir / f"{stamp}_{report['metadata']['provider']}.json",
        "latest": reports_dir / "latest.json",
        "markdown": reports_dir / "latest.md",
    }
    paths["report"].write_text(body)
    paths["latest"].write_text(body)
    paths["markdown"].write_text(render_markdown(report))
    if update_baseline:
        if report["metadata"]["subset"]:
            raise EvaluationError("Refusing to store a subset run as the baseline")
        paths["baseline"] = baseline_path(reports_dir, report["metadata"]["provider"])
        paths["baseline"].parent.mkdir(parents=True, exist_ok=True)
        paths["baseline"].write_text(body)
    return paths


def load_latest_report(reports_dir: Path) -> dict[str, Any] | None:
    path = reports_dir / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_evaluation(
    settings: Settings,
    *,
    provider_name: str = "mock",
    dataset_dir: Path | None = None,
    update_baseline: bool = False,
    case_ids: list[str] | None = None,
    audit: AuditStore | None = None,
    baseline_file: Path | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir or settings.golden_dir
    try:
        provider = build_provider(settings, provider_name)
    except (ProviderError, ValueError) as exc:
        raise EvaluationError(str(exc)) from exc
    processor = CaseProcessor(provider, load_policy(settings.policy_path), audit or AuditStore(settings.db_path), max_pdf_pages=settings.max_pdf_pages, max_text_chars=settings.max_text_chars)
    base_path = baseline_file or baseline_path(settings.reports_dir, provider.name)
    baseline = json.loads(base_path.read_text()) if base_path.exists() else None
    report = evaluate(processor, dataset_dir, load_thresholds(dataset_dir), baseline, case_ids)
    report["output_paths"] = {k: str(v) for k, v in write_report(report, settings.reports_dir, update_baseline).items()}
    return report
