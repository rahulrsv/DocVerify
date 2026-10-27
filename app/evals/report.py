"""Human-readable Markdown summary of an evaluation report."""

from typing import Any


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    meta, m, cmp = report["metadata"], report["metrics"], report["comparison"]
    ext, cls, e2e, gold, der = m["extraction"], m["classification"], m["policy_end_to_end"], m["policy_gold_extraction"], m["derived"]
    out = [
        f"# IncomeLens evaluation — {meta['eval_run_id']}",
        "",
        f"> **{meta['mode_label']}**",
        "",
        f"- Created: {meta['created_at']}  |  Duration: {meta['duration_s']}s",
        f"- Provider/model: `{meta['provider']}` / `{meta['model']}`  |  Prompt: `{meta['prompt_version']}`",
        f"- Schema: `{meta['schema_version']}`  |  Policy: `{meta['policy_version']}`",
        f"- Dataset: `{meta['dataset_version']}` ({meta['case_count']} cases, sha256 `{meta['dataset_sha256'][:12]}`){'' if meta['dataset_integrity_ok'] else '  **WARNING: dataset hash does not match manifest**'}",
        f"- Git commit: `{meta.get('git_commit') or 'n/a'}`",
        "",
        f"## Result: {'PASS' if report['summary']['thresholds_passed'] else 'FAIL'}",
        "",
        "| Threshold | Value | Bound | Status |",
        "|---|---|---|---|",
    ]
    for t in report["thresholds"]:
        bound = f">= {t['min']}" if "min" in t else f"<= {t['max']}"
        out.append(f"| {t['metric']} | {t['value']} | {bound} | {'pass' if t['passed'] else '**FAIL**'} |")
    out += [
        "",
        "## Extraction",
        "",
        f"- Overall field accuracy: {_pct(ext.get('overall_field_accuracy'))} ({ext.get('field_rows')} document-field pairs)",
        f"- Field-weighted score: {_pct(ext.get('field_weighted_score'))}",
        f"- Critical-field error rate: {_pct(ext.get('critical_field_error_rate'))}",
        f"- Missing-field detection rate: {_pct(ext.get('missing_field_detection_rate'))}  |  Unsupported value rate: {_pct(ext.get('unsupported_value_rate'))}",
        f"- Money exact-match rate: {_pct(ext.get('money_exact_match_rate'))}  |  Money MAE: {ext.get('money_mean_abs_error')}",
        f"- Deposit precision / recall: {_pct(ext.get('deposit_precision'))} / {_pct(ext.get('deposit_recall'))}",
        f"- Classification accuracy: {_pct(cls.get('accuracy'))}  |  Macro F1: {cls.get('macro_f1')}",
        "",
        "| Field | Rows | Accuracy |",
        "|---|---|---|",
    ]
    out += [f"| {r['field']} | {r['rows']} | {_pct(r['accuracy'])} |" for r in ext.get("per_field", [])]
    out += ["", "| Document type | Rows | Accuracy |", "|---|---|---|"]
    out += [f"| {r['doc_type']} | {r['rows']} | {_pct(r['accuracy'])} |" for r in ext.get("per_document_type", [])]
    out += [
        "",
        "## Policy",
        "",
        "| Metric | End-to-end | Gold extractions |",
        "|---|---|---|",
    ]
    for key in ("outcome_agreement", "rule_level_agreement", "false_approval_rate", "incorrect_escalation_rate", "missing_evidence_detection_rate"):
        out.append(f"| {key} | {_pct(e2e.get(key))} | {_pct(gold.get(key))} |")
    out += [
        "",
        "End-to-end runs the real extractor; gold extractions feed labeled fields into calculations and policy to isolate rule logic.",
        "",
        f"Derived: recurring income agreement {_pct(der.get('recurring_income_agreement'))}, income pattern agreement {_pct(der.get('income_pattern_agreement'))}.",
        "",
        "## Baseline comparison",
        "",
    ]
    if not cmp["baseline_found"]:
        out.append("No baseline found. Run with `--update-baseline` to store this run as the baseline.")
    else:
        out.append(f"Baseline: `{cmp['baseline_eval_run_id']}` ({cmp['baseline_created_at']}). Regressions: **{'YES' if cmp['has_regressions'] else 'none'}**")
        if cmp["metadata_changes"]:
            out.append("")
            out.append("Changed since baseline: " + ", ".join(f"{k} `{v['baseline']}` -> `{v['current']}`" for k, v in cmp["metadata_changes"].items()))
        out += ["", "| Metric | Baseline | Current | Delta |", "|---|---|---|---|"]
        out += [f"| {d['metric']} | {d['baseline']} | {d['current']} | {d['delta']:+}{' **regressed**' if d['regressed'] else ''} |" for d in cmp["metric_deltas"]]
        for c in cmp["newly_failing_cases"]:
            out.append(f"- Newly failing: **{c['case_id']}** expected {c['expected']}, was {c['baseline_actual']}, now {c['current_actual']}")
        for f in cmp["field_regressions"]:
            out.append(f"- Field regression: **{f['field']}** {_pct(f['baseline'])} -> {_pct(f['current'])}")
        if cmp["newly_passing_cases"]:
            out.append(f"- Newly passing: {', '.join(cmp['newly_passing_cases'])}")
    failing = [c for c in report["cases"] if not c["passed"]]
    out += ["", f"## Failing cases ({len(failing)})", ""]
    if failing:
        out += ["| Case | Category | Expected | Actual | Rule diff | Field errors |", "|---|---|---|---|---|---|"]
        for c in failing:
            out.append(f"| {c['case_id']} | {c['category']} | {c['expected_outcome']} | {c['actual_outcome']} | {c['rule_diff'] or '-'} | {len(c['field_errors'])} |")
    else:
        out.append("None.")
    return "\n".join(out) + "\n"
