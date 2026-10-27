"""Golden-set regression gate in mock mode. Fails CI if thresholds or the stored baseline regress."""

import dataclasses
import json

from app.core.config import PROJECT_ROOT, Settings
from app.evals.dataset import load_dataset
from app.evals.run import main as eval_main
from app.evals.service import run_evaluation

GOLDEN = PROJECT_ROOT / "data" / "golden"
BASELINE = PROJECT_ROOT / "reports" / "baselines" / "mock.json"


def test_golden_dataset_integrity():
    ds = load_dataset(GOLDEN)
    assert ds.integrity_ok, "cases.jsonl changed without regenerating manifest.json"
    assert len(ds.cases) >= 40
    assert len({c.case_id for c in ds.cases}) == len(ds.cases)
    for case in ds.cases:
        assert case.reason and case.expected_outcome in {"APPROVE", "REFER_FOR_REVIEW", "INSUFFICIENT_EVIDENCE"}
        assert all(d.path.exists() for d in case.documents), case.case_id
    categories = {c.category for c in ds.cases}
    assert {"clean", "missing_fields", "conflicting_information", "implausible_dates", "duplicate", "variable_income", "malformed", "ambiguous", "misleading_text", "prompt_injection"} <= categories


def test_mock_golden_run_meets_thresholds_and_baseline(tmp_path):
    settings = dataclasses.replace(Settings(), db_path=tmp_path / "eval.db", reports_dir=tmp_path / "reports")
    report = run_evaluation(settings, provider_name="mock", baseline_file=BASELINE)
    assert report["summary"]["thresholds_passed"], report["summary"]["failed_thresholds"]
    assert report["metrics"]["policy_gold_extraction"]["outcome_agreement"] == 1.0
    assert report["metrics"]["policy_end_to_end"]["false_approval_rate"] == 0.0
    assert report["metadata"]["mode"] == "mock"
    if BASELINE.exists():
        assert not report["comparison"]["has_regressions"], report["comparison"]


def test_cli_exits_nonzero_when_threshold_fails(tmp_path, monkeypatch):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "cases.jsonl").write_text((GOLDEN / "cases.jsonl").read_text())
    (golden / "eval_thresholds.json").write_text(json.dumps({"thresholds": {"policy_end_to_end.outcome_agreement": {"min": 1.0}}}))
    monkeypatch.setenv("INCOMELENS_DB_PATH", str(tmp_path / "e.db"))
    from app.evals import dataset

    monkeypatch.setattr(dataset, "DOCUMENTS_SUBDIR", GOLDEN.parent / "synthetic_documents")
    code = eval_main(["--dataset", str(golden), "--reports-dir", str(tmp_path / "r"), "--baseline", str(tmp_path / "none.json")])
    assert code == 1  # GC050 is a known mock-extractor failure, so 100% agreement cannot be met


def test_cli_regression_exit_code(tmp_path):
    reports = tmp_path / "r"
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    if base is None:
        return
    # Pretend the baseline passed every case, so the known GC050 failure appears as a new regression.
    for c in base["cases"]:
        c["passed"] = True
    fake = tmp_path / "baseline.json"
    fake.write_text(json.dumps(base))
    code = eval_main(["--dataset", str(GOLDEN), "--reports-dir", str(reports), "--audit-db", str(tmp_path / "e.db"), "--baseline", str(fake), "--fail-on-regression"])
    assert code == 2
