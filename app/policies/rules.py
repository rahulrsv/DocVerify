"""Deterministic demonstration policy rules.

Each rule is a plain function over typed inputs that returns (outcome, explanation,
evidence). Rules read validated facts and derived calculations only. They never read
model recommendations, and the model cannot change them. These are demonstration rules,
not real underwriting criteria.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.models.case import CaseCalculations, DocumentRecord, Finding
from app.models.common import DocumentType, Outcome, RuleOutcome, Severity
from app.models.extraction import DocumentExtraction
from app.policies.config import PolicyConfig


@dataclass(frozen=True)
class PolicyContext:
    records: list[DocumentRecord]
    extractions: list[DocumentExtraction]
    calculations: CaseCalculations
    as_of_date: date

    def classified(self) -> list[DocumentExtraction]:
        return [e for e in self.extractions if e.document_type is not DocumentType.UNKNOWN]


RuleReturn = tuple[RuleOutcome, str, list[dict[str, Any]]]
RuleFn = Callable[[PolicyContext, PolicyConfig], RuleReturn]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    version: str
    title: str
    fail_effect: Outcome
    severity: Severity
    fn: RuleFn


RULES: list[RuleSpec] = []


def rule(rule_id: str, version: str, title: str, fail_effect: Outcome, severity: Severity) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        RULES.append(RuleSpec(rule_id, version, title, fail_effect, severity, fn))
        return fn

    return register


def _evidence(findings: list[Finding]) -> list[dict[str, Any]]:
    return [{"finding": f.code, "message": f.message, "document_ids": f.document_ids, "fields": f.fields} for f in findings]


def _findings_rule(ctx: PolicyContext, codes: tuple[str, ...], ok: str) -> RuleReturn:
    hits = ctx.calculations.by_code(*codes)
    if hits:
        return RuleOutcome.FAIL, "; ".join(f.message for f in hits), _evidence(hits)
    return RuleOutcome.PASS, ok, []


REFER, INSUFFICIENT = Outcome.REFER_FOR_REVIEW, Outcome.INSUFFICIENT_EVIDENCE


@rule("R001_REQUIRED_DOCUMENTS", "1.0", "Required income evidence is present", INSUFFICIENT, Severity.CRITICAL)
def required_documents(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    present = {e.document_type for e in ctx.classified()}
    missing = [t.value for t in cfg.required_document_types if t not in present]
    if missing:
        return RuleOutcome.FAIL, f"Missing required document type(s): {', '.join(missing)}", [{"missing_document_types": missing}]
    return RuleOutcome.PASS, f"All required document types present: {', '.join(t.value for t in cfg.required_document_types)}", []


@rule("R002_CRITICAL_FIELDS_PRESENT", "1.0", "Critical fields are present and grounded", INSUFFICIENT, Severity.CRITICAL)
def critical_fields(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    if not ctx.classified():
        return RuleOutcome.NOT_APPLICABLE, "No classified documents to check", []
    return _findings_rule(ctx, ("MISSING_CRITICAL_FIELD",), "Every classified document has its critical fields")


@rule("R003_DOCUMENTS_READABLE", "1.0", "All documents are readable and of a supported type", REFER, Severity.WARNING)
def documents_readable(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    return _findings_rule(ctx, ("DOCUMENT_UNREADABLE", "UNCLASSIFIED_DOCUMENT"), "All documents were read and classified")


@rule("R004_NO_DUPLICATE_DOCUMENTS", "1.0", "No duplicate documents were submitted", REFER, Severity.WARNING)
def no_duplicates(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    return _findings_rule(ctx, ("DUPLICATE_DOCUMENT",), "No duplicate documents detected")


@rule("R005_DATES_AND_AMOUNTS_PLAUSIBLE", "1.0", "Pay dates, periods and amounts are plausible", REFER, Severity.WARNING)
def dates_plausible(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    if not any(e.document_type in (DocumentType.PAY_STUB, DocumentType.EMPLOYMENT_LETTER) for e in ctx.classified()):
        return RuleOutcome.NOT_APPLICABLE, "No pay stub or employment letter to check", []
    codes = ("FUTURE_DATE", "INVALID_PAY_PERIOD", "PAY_PERIOD_FREQUENCY_MISMATCH", "IMPLAUSIBLE_PAY_DATE", "YTD_LESS_THAN_GROSS", "NET_EXCEEDS_GROSS", "PAY_FREQUENCY_CONFLICT")
    return _findings_rule(ctx, codes, "Dates, pay periods and amounts are internally consistent")


@rule("R006_IDENTITY_CONSISTENT", "1.0", "Employee identity is consistent across documents", REFER, Severity.CRITICAL)
def identity_consistent(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    if sum(1 for e in ctx.classified() if e.employee_name.value) < 2:
        return RuleOutcome.NOT_APPLICABLE, "Fewer than two documents state an employee name", []
    return _findings_rule(ctx, ("EMPLOYEE_NAME_MISMATCH",), "All documents name the same person")


@rule("R007_EMPLOYER_CONSISTENT", "1.0", "Employer is consistent across documents", REFER, Severity.CRITICAL)
def employer_consistent(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    if sum(1 for e in ctx.classified() if e.employer_name.value) < 2:
        return RuleOutcome.NOT_APPLICABLE, "Fewer than two documents state an employer", []
    return _findings_rule(ctx, ("EMPLOYER_MISMATCH",), "All documents name the same employer")


@rule("R008_SALARY_CONSISTENT", "1.0", "Documented salary agrees with pay stub income", REFER, Severity.WARNING)
def salary_consistent(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    conflicts = ctx.calculations.by_code("CONFLICTING_SALARY_DOCUMENTS")
    if conflicts:
        return RuleOutcome.FAIL, conflicts[0].message, _evidence(conflicts)
    variance = ctx.calculations.salary_variance_pct
    if variance.value is None:
        return RuleOutcome.NOT_APPLICABLE, "No documented salary and recurring pay stub income to compare", []
    tolerance = cfg.thresholds.salary_tolerance_pct
    evidence = [{"salary_variance_pct": str(variance.value), "tolerance_pct": str(tolerance), "inputs": [i.model_dump() for i in variance.inputs]}]
    if variance.value > tolerance:
        return (
            RuleOutcome.FAIL,
            f"Annualized pay stub income {ctx.calculations.recurring_annual_income.value} differs from documented salary "
            f"{ctx.calculations.documented_annual_salary.value} by {variance.value}% (tolerance {tolerance}%)",
            evidence,
        )
    return RuleOutcome.PASS, f"Salary variance {variance.value}% is within {tolerance}% tolerance", evidence


@rule("R009_INCOME_RECURRING", "1.0", "Income is recurring and can be annualized", REFER, Severity.WARNING)
def income_recurring(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    pattern = ctx.calculations.income_pattern
    if pattern == "UNKNOWN":
        return RuleOutcome.NOT_APPLICABLE, "Income pattern not determinable (missing pay stub gross or frequency)", []
    if pattern == "VARIABLE":
        hits = ctx.calculations.by_code("VARIABLE_INCOME", "IRREGULAR_PAY_FREQUENCY")
        return RuleOutcome.FAIL, "Variable or irregular income requires manual review: " + "; ".join(f.message for f in hits), _evidence(hits)
    return RuleOutcome.PASS, "Pay stub income is recurring at a supported frequency", []


@rule("R010_INCOME_THRESHOLD", "1.0", "Recurring income meets the demonstration threshold", REFER, Severity.WARNING)
def income_threshold(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    income = ctx.calculations.recurring_annual_income
    if income.value is None:
        return RuleOutcome.NOT_APPLICABLE, "No recurring annual income was calculated", []
    threshold = cfg.thresholds.min_recurring_annual_income
    evidence = [{"recurring_annual_income": str(income.value), "threshold": str(threshold), "method": income.method}]
    if income.value < threshold:
        return RuleOutcome.FAIL, f"Recurring annual income {income.value} is below the demo threshold {threshold}", evidence
    return RuleOutcome.PASS, f"Recurring annual income {income.value} meets the demo threshold {threshold}", evidence


@rule("R011_EMPLOYMENT_ACTIVE", "1.0", "Employment is documented as active", REFER, Severity.CRITICAL)
def employment_active(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    if not any(e.employment_status.value for e in ctx.classified() if e.document_type is DocumentType.EMPLOYMENT_LETTER):
        return RuleOutcome.NOT_APPLICABLE, "No employment status documented", []
    return _findings_rule(ctx, ("EMPLOYMENT_NOT_ACTIVE",), "Employment letter documents active employment")


@rule("R012_DOCUMENT_RECENCY", "1.0", "Pay stub evidence is recent", INSUFFICIENT, Severity.WARNING)
def document_recency(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    age = ctx.calculations.latest_pay_stub_age_days
    if age.value is None:
        return RuleOutcome.NOT_APPLICABLE, "No dated pay stub", []
    limit = cfg.thresholds.max_pay_stub_age_days
    evidence = [{"latest_pay_stub_age_days": str(age.value), "max_age_days": limit}]
    if age.value > limit:
        return RuleOutcome.FAIL, f"Most recent pay stub is {age.value} days old (limit {limit}); a recent pay stub is required", evidence
    return RuleOutcome.PASS, f"Most recent pay stub is {age.value} days old (limit {limit})", evidence


@rule("R013_NO_MANIPULATION_SIGNALS", "1.0", "No document attempts to manipulate automated processing", REFER, Severity.CRITICAL)
def no_manipulation(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    hits = ctx.calculations.by_code("PROMPT_INJECTION_SUSPECTED")
    # A model self-report can only ever escalate to review, never relax a rule.
    flagged = [e.document_id for e in ctx.extractions if e.model_flagged_suspicious]
    if hits or flagged:
        evidence = _evidence(hits) + ([{"model_flagged_documents": flagged}] if flagged else [])
        message = "; ".join(f.message for f in hits) or f"Extraction model flagged instruction-like content in {', '.join(flagged)}"
        return RuleOutcome.FAIL, message, evidence
    return RuleOutcome.PASS, "No instruction-like content detected", []


@rule("R014_CURRENCY_SUPPORTED", "1.0", "Amounts are in a supported, consistent currency", REFER, Severity.WARNING)
def currency_supported(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    found = {e.currency.value for e in ctx.classified() if e.currency.value}
    if not found:
        return RuleOutcome.NOT_APPLICABLE, "No currency stated", []
    unsupported = sorted(c for c in found if c not in cfg.allowed_currencies)
    mismatch = ctx.calculations.by_code("CURRENCY_MISMATCH")
    if unsupported or mismatch:
        parts = ([f"Unsupported currency: {', '.join(unsupported)} (allowed: {', '.join(cfg.allowed_currencies)})"] if unsupported else []) + [m.message for m in mismatch]
        return RuleOutcome.FAIL, "; ".join(parts), [{"currencies": sorted(found)}] + _evidence(mismatch)
    return RuleOutcome.PASS, f"All amounts in {', '.join(sorted(found))}", []


@rule("R015_DEPOSITS_CORROBORATE", "1.0", "Bank deposits corroborate pay stub net pay", REFER, Severity.WARNING)
def deposits_corroborate(ctx: PolicyContext, cfg: PolicyConfig) -> RuleReturn:
    matched = ctx.calculations.matched_payroll_deposits
    if matched.value is None:
        return RuleOutcome.NOT_APPLICABLE, "No bank statement with a comparable pay stub net pay", []
    if ctx.calculations.has("DEPOSITS_DO_NOT_CORROBORATE"):
        return _findings_rule(ctx, ("DEPOSITS_DO_NOT_CORROBORATE",), "")
    return RuleOutcome.PASS, f"{matched.value} payroll deposit(s) match pay stub net pay", [{"matched_payroll_deposits": str(matched.value)}]
