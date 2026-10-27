"""Regression evaluation CLI.

    python -m app.evals.run --dataset data/golden                 # mock provider, compare to baseline
    python -m app.evals.run --dataset data/golden --update-baseline
    python -m app.evals.run --dataset data/golden --provider openai  # needs OPENAI_API_KEY

Exit codes: 0 = thresholds passed, 1 = a critical threshold failed,
2 = regression vs baseline (only with --fail-on-regression), 3 = configuration error.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evals.service import EvaluationError, run_evaluation


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the IncomeLens golden-set evaluation")
    parser.add_argument("--dataset", type=Path, default=None, help="Golden dataset directory (default: data/golden)")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline report (default: reports/baselines/<provider>.json)")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--audit-db", type=Path, default=None, help="SQLite file for evaluation runs (default: INCOMELENS_DB_PATH)")
    parser.add_argument("--cases", nargs="*", default=None, help="Optional subset of case IDs")
    parser.add_argument("--update-baseline", action="store_true", help="Store this run as the new baseline")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit 2 when regressions vs baseline are found")
    args = parser.parse_args(argv)

    settings = get_settings()
    overrides = {k: v for k, v in {"reports_dir": args.reports_dir, "db_path": args.audit_db}.items() if v is not None}
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    configure_logging("WARNING")

    try:
        report = run_evaluation(settings, provider_name=args.provider, dataset_dir=args.dataset, update_baseline=args.update_baseline, case_ids=args.cases, baseline_file=args.baseline)
    except (EvaluationError, FileNotFoundError) as exc:
        print(f"Evaluation configuration error: {exc}", file=sys.stderr)
        return 3

    s, m, cmp = report["summary"], report["metrics"], report["comparison"]
    print("=" * 78)
    print(report["metadata"]["mode_label"])
    print("=" * 78)
    meta = report["metadata"]
    print(f"run {meta['eval_run_id']} | dataset {meta['dataset_version']} ({meta['case_count']} cases) | policy {meta['policy_version']} | prompt {meta['prompt_version']}")
    print(f"cases passed (outcome + rule set match): {s['cases_passed']}/{s['cases']}")
    print("\nEXTRACTION")
    ext = m["extraction"]
    print(f"  overall field accuracy      {_pct(ext.get('overall_field_accuracy'))}   field-weighted score {_pct(ext.get('field_weighted_score'))}")
    print(f"  critical-field error rate   {_pct(ext.get('critical_field_error_rate'))}   missing-field detection {_pct(ext.get('missing_field_detection_rate'))}")
    print(f"  money exact match           {_pct(ext.get('money_exact_match_rate'))}   classification macro F1 {m['classification']['macro_f1']}")
    print("\nPOLICY                        end-to-end   gold-extraction")
    for key in ("outcome_agreement", "rule_level_agreement", "false_approval_rate", "incorrect_escalation_rate", "missing_evidence_detection_rate"):
        print(f"  {key:<28}{_pct(m['policy_end_to_end'][key]):>10}   {_pct(m['policy_gold_extraction'][key]):>10}")
    print("\nTHRESHOLDS")
    for t in report["thresholds"]:
        bound = f">= {t['min']}" if "min" in t else f"<= {t['max']}"
        print(f"  [{'PASS' if t['passed'] else 'FAIL'}] {t['metric']} = {t['value']} ({bound})")
    print("\nBASELINE")
    if not cmp["baseline_found"]:
        print("  no baseline found (use --update-baseline)")
    else:
        print(f"  vs {cmp['baseline_eval_run_id']}: {'REGRESSIONS FOUND' if cmp['has_regressions'] else 'no regressions'}")
        for d in cmp["metric_deltas"]:
            if d["regressed"] or d["delta"]:
                print(f"  {'!!' if d['regressed'] else '  '} {d['metric']}: {d['baseline']} -> {d['current']} ({d['delta']:+})")
        for c in cmp["newly_failing_cases"]:
            print(f"  !! newly failing {c['case_id']}: expected {c['expected']}, now {c['current_actual']} {c['rule_diff']}")
        for f in cmp["field_regressions"]:
            print(f"  !! field {f['field']}: {_pct(f['baseline'])} -> {_pct(f['current'])}")
    failing = [c for c in report["cases"] if not c["passed"]]
    if failing:
        print("\nFAILING CASES")
        for c in failing:
            print(f"  {c['case_id']} [{c['category']}] expected {c['expected_outcome']} got {c['actual_outcome']} {c['rule_diff']} ({len(c['field_errors'])} field errors)")
    print("\nreports: " + ", ".join(f"{k}={v}" for k, v in report["output_paths"].items()))

    if not s["thresholds_passed"]:
        return 1
    if args.fail_on_regression and cmp["has_regressions"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
