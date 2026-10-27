import dataclasses
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.audit.store import AuditStore
from app.core.config import PROJECT_ROOT, Settings
from app.core.pipeline import CaseProcessor
from app.extraction.mock_provider import RuleBasedExtractionProvider
from app.main import create_app
from app.models.common import DocumentType, ExtractionStatus, PayFrequency
from app.models.extraction import DocumentExtraction, ExtractedField
from app.policies.config import load_policy

DOCS = PROJECT_ROOT / "data" / "synthetic_documents"
POLICY_PATH = PROJECT_ROOT / "data" / "policies" / "demo_policy_v1.json"
AS_OF = date(2026, 10, 15)


@pytest.fixture
def policy():
    return load_policy(POLICY_PATH)


@pytest.fixture
def audit(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "audit.db")


@pytest.fixture
def processor(audit, policy) -> CaseProcessor:
    return CaseProcessor(RuleBasedExtractionProvider(), policy, audit)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return dataclasses.replace(
        Settings(),
        provider="mock",
        db_path=tmp_path / "api.db",
        reports_dir=tmp_path / "reports",
        max_upload_bytes=200_000,
        openai_api_key=None,
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def doc_bytes(case_id: str, filename: str) -> bytes:
    return (DOCS / case_id / filename).read_bytes()


def make_extraction(document_id: str, doc_type: DocumentType, **values) -> DocumentExtraction:
    """Build a canonical extraction directly, for calculation and policy unit tests."""
    names = [
        "employee_name", "employer_name", "pay_period_start", "pay_period_end", "pay_date", "gross_pay", "net_pay",
        "pay_frequency", "ytd_gross", "employment_start_date", "employment_status", "base_annual_salary", "currency",
    ]
    fields = {n: ExtractedField(value=values.get(n), evidence="test" if values.get(n) is not None else None, source_document_id=document_id) for n in names}
    return DocumentExtraction(
        document_id=document_id,
        document_type=doc_type,
        deposits=values.get("deposits", []),
        status=ExtractionStatus.SUCCESS,
        missing_critical_fields=values.get("missing", []),
        **fields,
    )


def stub(document_id="D1", gross="3076.92", freq=PayFrequency.BIWEEKLY, start=date(2026, 9, 14), end=date(2026, 9, 27), paid=date(2026, 10, 2), **kw):
    values = dict(
        employee_name="Jordan Avery", employer_name="Northwind Logistics Inc.", pay_period_start=start, pay_period_end=end,
        pay_date=paid, gross_pay=Decimal(gross) if gross else None, net_pay=Decimal("2284.11"), pay_frequency=freq,
        ytd_gross=Decimal("60000.00"), currency="CAD",
    )
    values.update(kw)
    return make_extraction(document_id, DocumentType.PAY_STUB, **values)


def letter(document_id="D2", salary="80000.00", **kw):
    from app.models.common import EmploymentStatus

    values = dict(
        employee_name="Jordan Avery", employer_name="Northwind Logistics Inc.", employment_start_date=date(2019, 4, 8),
        employment_status=EmploymentStatus.FULL_TIME, base_annual_salary=Decimal(salary) if salary else None, currency="CAD",
    )
    values.update(kw)
    return make_extraction(document_id, DocumentType.EMPLOYMENT_LETTER, **values)


def record(document_id: str, sha: str | None = None, text_sha: str | None = None, **kw):
    from app.models.case import DocumentRecord

    return DocumentRecord(document_id=document_id, filename=f"{document_id}.txt", media_type="text/plain", sha256=sha or f"sha-{document_id}", text_sha256=text_sha or f"text-{document_id}", **kw)
