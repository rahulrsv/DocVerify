"""Case-level models: documents, derived metrics, rule results and the final result."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.models.common import DISCLAIMER, Outcome, RuleOutcome, Severity
from app.models.extraction import DocumentExtraction


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    media_type: str
    sha256: str = Field(description="Hash of the uploaded bytes.")
    text_sha256: str | None = Field(default=None, description="Hash of whitespace-normalized extracted text.")
    page_count: int | None = None
    char_count: int = 0
    load_error: str | None = None
    injection_signals: list[str] = Field(default_factory=list)


class InputRef(BaseModel):
    document_id: str
    field: str
    value: str | None


class DerivedValue(BaseModel):
    """A value computed deterministically from extracted facts. Never produced by a model."""

    name: str
    value: Decimal | None
    method: str
    inputs: list[InputRef] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """A factual observation from validation or calculation. Policy decides what it means."""

    code: str
    severity: Severity
    message: str
    document_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)


class CaseCalculations(BaseModel):
    annualized_by_pay_stub: list[DerivedValue] = Field(default_factory=list)
    recurring_annual_income: DerivedValue
    documented_annual_salary: DerivedValue
    salary_variance_pct: DerivedValue
    gross_pay_variation_pct: DerivedValue
    latest_pay_stub_age_days: DerivedValue
    matched_payroll_deposits: DerivedValue
    income_pattern: str = Field(description="RECURRING, VARIABLE or UNKNOWN")
    findings: list[Finding] = Field(default_factory=list)

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    def by_code(self, *codes: str) -> list[Finding]:
        return [f for f in self.findings if f.code in codes]


class PolicyCitation(BaseModel):
    """Handbook passage retrieved for a failed rule. Explanatory only; it never affects the outcome."""

    passage_id: str
    title: str
    excerpt: str
    score: float
    retriever: str = Field(description="Handbook version and retrieval backend, e.g. demo-handbook-1.0/bm25.")
    matches_rule: bool = Field(description="True when the retrieved passage is the one written for this rule.")


class RuleResult(BaseModel):
    rule_id: str
    rule_version: str
    title: str
    outcome: RuleOutcome
    fail_effect: Outcome
    severity: Severity
    explanation: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citation: PolicyCitation | None = None


class Decision(BaseModel):
    outcome: Outcome
    policy_version: str
    summary: str
    reasons: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class ProcessingMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    policy_version: str
    retrieval_version: str | None = None
    extraction_attempts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    run_id: str
    case_id: str
    processed_at: datetime
    as_of_date: date
    input_fingerprint: str
    documents: list[DocumentRecord]
    extractions: list[DocumentExtraction]
    calculations: CaseCalculations
    rule_results: list[RuleResult]
    decision: Decision
    metadata: ProcessingMetadata
    duration_ms: int
    source: str = "api"
    evaluation_run_id: str | None = None
    idempotent_replay: bool = False
