"""Golden dataset loading and conversion of labels into canonical extractions."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.extraction.normalize import parse_money
from app.models.common import CRITICAL_FIELDS_BY_TYPE, DATE_FIELDS, FIELDS_BY_TYPE, MONEY_FIELDS, SCALAR_FIELDS, DocumentType, EmploymentStatus, ExtractionStatus, PayFrequency
from app.models.extraction import BankDeposit, DocumentExtraction, ExtractedField

DOCUMENTS_SUBDIR = Path("..") / "synthetic_documents"


@dataclass(frozen=True)
class GoldenDocument:
    path: Path
    expected: dict[str, Any]

    @property
    def expected_type(self) -> DocumentType:
        return DocumentType(self.expected["document_type"])


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    category: str
    description: str
    as_of_date: date
    documents: list[GoldenDocument]
    expected_derived: dict[str, Any]
    expected_outcome: str
    expected_failed_rules: list[str]
    reason: str


@dataclass(frozen=True)
class GoldenDataset:
    version: str
    sha256: str
    manifest_sha256: str | None
    cases: list[GoldenCase]

    @property
    def integrity_ok(self) -> bool:
        return self.manifest_sha256 is None or self.manifest_sha256 == self.sha256


def load_dataset(dataset_dir: Path, documents_dir: Path | None = None) -> GoldenDataset:
    cases_path = dataset_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {cases_path}")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    docs_root = documents_dir or (dataset_dir / DOCUMENTS_SUBDIR).resolve()
    cases = []
    for line in cases_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cases.append(
            GoldenCase(
                case_id=item["case_id"],
                category=item["category"],
                description=item["description"],
                as_of_date=date.fromisoformat(item["as_of_date"]),
                documents=[GoldenDocument(path=docs_root / d["path"], expected=d["expected"]) for d in item["documents"]],
                expected_derived=item.get("expected_derived", {}),
                expected_outcome=item["expected_outcome"],
                expected_failed_rules=item.get("expected_failed_rules", []),
                reason=item["reason"],
            )
        )
    return GoldenDataset(
        version=manifest.get("dataset_version", "unversioned"),
        sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        manifest_sha256=manifest.get("cases_sha256"),
        cases=cases,
    )


def _parse_label(field: str, value: str | None) -> Any:
    if value is None:
        return None
    if field in MONEY_FIELDS:
        return parse_money(value)
    if field in DATE_FIELDS:
        return date.fromisoformat(value)
    if field == "pay_frequency":
        return PayFrequency(value)
    if field == "employment_status":
        return EmploymentStatus(value)
    return value


def gold_extraction(document_id: str, expected: dict[str, Any]) -> DocumentExtraction:
    """Build the extraction a perfect extractor would return. Used to test policy logic in isolation."""
    doc_type = DocumentType(expected["document_type"])
    applicable = FIELDS_BY_TYPE[doc_type]
    fields = {
        name: ExtractedField(
            value=_parse_label(name, expected.get(name)) if name in applicable else None,
            evidence="[gold label]" if name in applicable and expected.get(name) is not None else None,
            source_document_id=document_id,
        )
        for name in SCALAR_FIELDS
    }
    deposits = [
        BankDeposit(date=date.fromisoformat(d["date"]), amount=parse_money(d["amount"]), evidence="[gold label]", source_document_id=document_id)
        for d in expected.get("deposits", [])
    ]
    missing = [f for f in CRITICAL_FIELDS_BY_TYPE[doc_type] if fields[f].value is None]
    status = ExtractionStatus.SUCCESS if not missing and doc_type is not DocumentType.UNKNOWN else ExtractionStatus.PARTIAL
    return DocumentExtraction(document_id=document_id, document_type=doc_type, deposits=deposits, status=status, missing_critical_fields=missing, **fields)  # type: ignore[arg-type]
