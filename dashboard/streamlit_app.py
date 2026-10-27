"""IncomeLens AI demo dashboard. Talks to the FastAPI backend over HTTP."""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "synthetic_documents"
GOLDEN = ROOT / "data" / "golden" / "cases.jsonl"
OUTCOME_STYLE = {"APPROVE": ("green", "APPROVE"), "REFER_FOR_REVIEW": ("orange", "REFER FOR REVIEW"), "INSUFFICIENT_EVIDENCE": ("red", "INSUFFICIENT EVIDENCE")}
RULE_ICON = {"PASS": "✅", "FAIL": "❌", "NOT_APPLICABLE": "➖"}

st.set_page_config(page_title="IncomeLens AI", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach API at {API_URL}: {exc}")
        return None, None
    try:
        body = r.json()
    except ValueError:
        body = {"detail": r.text}
    return r.status_code, body


@st.cache_data
def golden_cases() -> dict[str, dict]:
    if not GOLDEN.exists():
        return {}
    return {c["case_id"]: c for c in (json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip())}


def outcome_badge(outcome: str) -> None:
    color, label = OUTCOME_STYLE.get(outcome, ("gray", outcome))
    st.markdown(f"### :{color}[{label}]")


# ----------------------------------------------------------------------- header

st.title("IncomeLens AI")
st.caption("Auditable income-document underwriting **simulation** · synthetic data only · outcomes are demonstration-rule results, not credit decisions")
code, health = api("GET", "/health")
if health and code == 200:
    st.caption(f"API `{API_URL}` · provider `{health['provider']}` / `{health['model']}` · prompt `{health['prompt_version']}` · policy `{health['policy_version']}` · schema `{health['schema_version']}`")
    if health["provider"] == "mock":
        st.info("Extraction is running in **mock mode** (deterministic rule-based extractor, not an LLM). Set INCOMELENS_PROVIDER=openai for live extraction.")
else:
    st.stop()

tab_process, tab_underwrite, tab_eval, tab_audit = st.tabs(["1 · Document processing", "2 · Underwriting simulation", "3 · Evaluation", "4 · Audit explorer"])

# ------------------------------------------------------------------ processing

with tab_process:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Load a golden sample case")
        cases = golden_cases()
        options = [""] + [f"{cid} · {c['description']}" for cid, c in cases.items()]
        choice = st.selectbox("Sample", options, index=0)
        if choice and st.button("Process sample", type="primary"):
            cid = choice.split(" · ")[0]
            case = cases[cid]
            files = [("files", (Path(d["path"]).name, (SAMPLES / d["path"]).read_bytes(), "application/octet-stream")) for d in case["documents"]]
            code, body = api("POST", "/documents/process", files=files, data={"case_id": cid, "as_of_date": case["as_of_date"], "force": "true"})
            if code in (200, 201):
                st.session_state["result"] = body
                st.session_state["expected"] = case
            else:
                st.error(f"HTTP {code}: {body.get('detail')}")
    with right:
        st.subheader("Or upload your own synthetic documents")
        uploads = st.file_uploader("PDF or TXT, one case may contain several related documents", type=["pdf", "txt"], accept_multiple_files=True)
        case_id = st.text_input("Case ID (optional)")
        as_of = st.date_input("As-of date", value=pd.Timestamp("2026-10-15"))
        if uploads and st.button("Process upload"):
            files = [("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
            data = {"as_of_date": str(as_of)}
            if case_id:
                data["case_id"] = case_id
            code, body = api("POST", "/documents/process", files=files, data=data)
            if code in (200, 201):
                st.session_state["result"] = body
                st.session_state.pop("expected", None)
            else:
                st.error(f"HTTP {code}: {body.get('detail')}")

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.markdown(f"**Case** `{result['case_id']}` · **run** `{result['run_id']}` · {result['duration_ms']} ms" + (" · idempotent replay" if result.get("idempotent_replay") else ""))
        for rec, ext in zip(result["documents"], result["extractions"], strict=True):
            with st.expander(f"📄 {rec['filename']} — {ext['document_type']} · {ext['status']}", expanded=True):
                if rec.get("load_error"):
                    st.error(rec["load_error"])
                if rec.get("injection_signals"):
                    st.warning(f"Instruction-like text detected: {', '.join(rec['injection_signals'])}. Treated as data; case will be referred.")
                rows = []
                for name, f in ext.items():
                    if isinstance(f, dict) and "source_document_id" in f and (f["value"] is not None or f["warnings"]):
                        rows.append({"field": name, "value": f["value"], "evidence": f["evidence"], "model confidence (uncalibrated)": f["model_confidence"], "warnings": "; ".join(f["warnings"])})
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                if ext["deposits"]:
                    st.markdown("**Payroll deposits**")
                    st.dataframe(pd.DataFrame(ext["deposits"])[["date", "amount", "description", "evidence"]], hide_index=True, width="stretch")
                if ext["missing_critical_fields"]:
                    st.error("Missing critical fields: " + ", ".join(ext["missing_critical_fields"]))
                other = [i for i in ext["issues"] if i["code"] != "MISSING_CRITICAL_FIELD"]
                if other:
                    st.markdown("**Validation issues**")
                    st.dataframe(pd.DataFrame(other)[["severity", "code", "field", "message"]], hide_index=True, width="stretch")

# --------------------------------------------------------------- underwriting

with tab_underwrite:
    result = st.session_state.get("result")
    if not result:
        st.info("Process a case first.")
    else:
        d, calc = result["decision"], result["calculations"]
        c1, c2 = st.columns([1, 2])
        with c1:
            outcome_badge(d["outcome"])
            st.caption(d["summary"])
            expected = st.session_state.get("expected")
            if expected:
                match = expected["expected_outcome"] == d["outcome"]
                st.markdown(f"Golden label: **{expected['expected_outcome']}** {'✅ match' if match else '❌ mismatch'}")
                st.caption(f"Label reason: {expected['reason']}")
            st.caption(d["disclaimer"])
        with c2:
            st.markdown("**Why this outcome**")
            if d["reasons"]:
                for reason in d["reasons"]:
                    st.markdown(f"- {reason}")
            else:
                st.markdown("- Every configured demonstration rule passed or was not applicable.")

        st.subheader("Calculated income (deterministic, no LLM arithmetic)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recurring annual income", calc["recurring_annual_income"]["value"] or "—")
        m2.metric("Documented salary", calc["documented_annual_salary"]["value"] or "—")
        m3.metric("Salary variance %", calc["salary_variance_pct"]["value"] or "—")
        m4.metric("Income pattern", calc["income_pattern"])
        derived = [calc[k] for k in ("recurring_annual_income", "documented_annual_salary", "salary_variance_pct", "gross_pay_variation_pct", "latest_pay_stub_age_days", "matched_payroll_deposits")]
        st.dataframe(
            pd.DataFrame([{"metric": v["name"], "value": v["value"], "method": v["method"], "inputs": ", ".join(f"{i['document_id']}.{i['field']}={i['value']}" for i in v["inputs"]), "assumptions": "; ".join(v["assumptions"])} for v in derived]),
            hide_index=True,
            width="stretch",
        )
        if calc["findings"]:
            st.markdown("**Findings**")
            st.dataframe(pd.DataFrame(calc["findings"])[["severity", "code", "message", "document_ids"]], hide_index=True, width="stretch")

        st.subheader(f"Policy rules · {d['policy_version']}")
        for r in result["rule_results"]:
            label = f"{RULE_ICON[r['outcome']]} {r['rule_id']} v{r['rule_version']} — {r['title']}"
            with st.expander(label, expanded=r["outcome"] == "FAIL"):
                st.markdown(f"**Outcome:** {r['outcome']} · **effect on fail:** {r['fail_effect']} · **severity:** {r['severity']}")
                st.markdown(r["explanation"])
                if r["evidence"]:
                    st.json(r["evidence"], expanded=False)

# ------------------------------------------------------------------ evaluation

with tab_eval:
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Run mock evaluation", help="Runs all golden cases with the deterministic mock extractor"):
            with st.spinner("Evaluating golden set..."):
                code, body = api("POST", "/evals/run", json={"provider": "mock"})
            if code != 200:
                st.error(f"HTTP {code}: {body.get('detail')}")
    code, report = api("GET", "/evals/latest")
    if code == 404:
        st.info("No evaluation yet. Run one with the button or `make eval`.")
    elif report:
        meta, m, cmp = report["metadata"], report["metrics"], report["comparison"]
        with c1:
            (st.warning if meta["mode"] == "mock" else st.success)(meta["mode_label"])
            st.caption(f"Run `{meta['eval_run_id']}` · {meta['created_at']} · dataset `{meta['dataset_version']}` ({meta['case_count']} cases) · policy `{meta['policy_version']}` · prompt `{meta['prompt_version']}`")
        s = report["summary"]
        k = st.columns(5)
        k[0].metric("Thresholds", "PASS" if s["thresholds_passed"] else "FAIL")
        k[1].metric("Cases passed", f"{s['cases_passed']}/{s['cases']}")
        k[2].metric("Field accuracy", f"{(m['extraction']['overall_field_accuracy'] or 0) * 100:.1f}%")
        k[3].metric("Outcome agreement (e2e)", f"{m['policy_end_to_end']['outcome_agreement'] * 100:.1f}%")
        k[4].metric("False approval rate", f"{m['policy_end_to_end']['false_approval_rate'] * 100:.1f}%")

        st.subheader("Critical thresholds")
        st.dataframe(pd.DataFrame(report["thresholds"]), hide_index=True, width="stretch")

        e1, e2 = st.columns(2)
        with e1:
            st.subheader("Per-field extraction accuracy")
            pf = pd.DataFrame(m["extraction"]["per_field"])
            st.bar_chart(pf.set_index("field")["accuracy"], horizontal=True)
            st.dataframe(pf, hide_index=True, width="stretch")
        with e2:
            st.subheader("Policy metrics")
            keys = ["outcome_agreement", "rule_level_agreement", "false_approval_rate", "incorrect_escalation_rate", "missing_evidence_detection_rate"]
            st.dataframe(pd.DataFrame({"metric": keys, "end_to_end": [m["policy_end_to_end"][x] for x in keys], "gold_extractions": [m["policy_gold_extraction"][x] for x in keys]}), hide_index=True, width="stretch")
            st.caption("Gold extractions feed labeled fields into calculations and rules, isolating policy logic from extraction error.")
            st.markdown("**Outcome confusion matrix (rows = expected)**")
            st.dataframe(pd.DataFrame(m["policy_end_to_end"]["confusion_matrix"]).T, width="stretch")

        st.subheader("Baseline comparison")
        if not cmp["baseline_found"]:
            st.info("No baseline stored. Run `make baseline`.")
        else:
            (st.error if cmp["has_regressions"] else st.success)(f"vs baseline {cmp['baseline_eval_run_id']}: {'REGRESSIONS FOUND' if cmp['has_regressions'] else 'no regressions'}")
            if cmp["metadata_changes"]:
                st.caption("Changed since baseline: " + ", ".join(f"{k}: {v['baseline']} → {v['current']}" for k, v in cmp["metadata_changes"].items()))
            st.dataframe(pd.DataFrame(cmp["metric_deltas"]), hide_index=True, width="stretch")
            if cmp["newly_failing_cases"]:
                st.error("Newly failing cases")
                st.dataframe(pd.DataFrame(cmp["newly_failing_cases"]), hide_index=True, width="stretch")

        st.subheader("Case results")
        only_fail = st.toggle("Show failing cases only", value=True)
        cases_df = pd.DataFrame(report["cases"])
        if only_fail:
            cases_df = cases_df[~cases_df["passed"]]
        st.dataframe(cases_df[["case_id", "category", "expected_outcome", "actual_outcome", "gold_extraction_outcome", "rule_diff", "field_accuracy", "passed"]], hide_index=True, width="stretch")
        errors = [dict(case_id=c["case_id"], **e) for c in report["cases"] for e in c["field_errors"]]
        if errors:
            with st.expander(f"Field-level errors ({len(errors)})"):
                st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")

# ----------------------------------------------------------------------- audit

with tab_audit:
    default = st.session_state.get("result", {}).get("case_id", "")
    query = st.text_input("Case ID", value=default, placeholder="e.g. GC044")
    code, recent = api("GET", "/cases", params={"limit": 20})
    if recent:
        st.caption("Recent cases: " + ", ".join(c["case_id"] for c in recent))
    if query:
        code, audit = api("GET", f"/cases/{query}/audit")
        if code == 404:
            st.warning("No history for that case.")
        elif code == 200:
            st.markdown(f"Hash chain verified: {'✅' if audit['chain_verified'] else '❌ TAMPERING DETECTED'} · {len(audit['runs'])} run(s) · {len(audit['events'])} event(s)")
            st.markdown("**Processing runs**")
            st.dataframe(pd.DataFrame(audit["runs"]), hide_index=True, width="stretch")
            st.markdown("**Events (chronological, append-only)**")
            for e in audit["events"]:
                with st.expander(f"{e['created_at']} · {e['event_type']} · run {e['run_id']}"):
                    st.json(e["payload"])
                    st.caption(f"event_hash {e['event_hash'][:16]}… · prev {e['prev_hash'][:16]}…")
            code, latest = api("GET", f"/cases/{query}")
            if code == 200:
                with st.expander("Latest full run record (model output, validation, rule results)"):
                    st.json(latest, expanded=False)
        else:
            st.error(f"HTTP {code}: {audit.get('detail') if audit else ''}")
