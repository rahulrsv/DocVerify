"""Deterministic income calculations and cross-document consistency checks.

No model is involved here. Every derived value carries its method, input references
and assumptions. This module reports *facts* (findings and numbers); the policy engine
decides what they mean for the outcome.

Assumptions (also documented in ARCHITECTURE.md):
* Periods per year: weekly 52, biweekly 26, semimonthly 24, monthly 12.
* Recurring annual income = most recent non-duplicate pay stub gross x periods per year,
  only when all pay stubs share a supported frequency and gross variation is within the
  configured variability threshold. Irregular or variable income is never annualized.
* Expected period length (inclusive days): weekly 7, biweekly 14, semimonthly 13-16,
  monthly 28-31, each with +/-1 day tolerance.
* A pay date may fall up to 3 days before and up to 15 days after the period end.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from app.extraction.normalize import to_iso
from app.models.case import CaseCalculations, DerivedValue, DocumentRecord, Finding, InputRef
from app.models.common import DocumentType, EmploymentStatus, PayFrequency, Severity
from app.models.extraction import DocumentExtraction

CENT = Decimal("0.01")
PERIODS_PER_YEAR: dict[PayFrequency, int] = {
    PayFrequency.WEEKLY: 52,
    PayFrequency.BIWEEKLY: 26,
    PayFrequency.SEMIMONTHLY: 24,
    PayFrequency.MONTHLY: 12,
}
PERIOD_LENGTH_DAYS: dict[PayFrequency, tuple[int, int]] = {
    PayFrequency.WEEKLY: (6, 8),
    PayFrequency.BIWEEKLY: (13, 15),
    PayFrequency.SEMIMONTHLY: (12, 17),
    PayFrequency.MONTHLY: (27, 32),
}
PAY_DATE_MAX_DAYS_BEFORE_END = 3
PAY_DATE_MAX_DAYS_AFTER_END = 15
_COMPANY_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "llc", "llp", "corp", "corporation", "co", "company"}


@dataclass(frozen=True)
class CalculationSettings:
    variable_income_threshold_pct: Decimal = Decimal("10")


# ------------------------------------------------------------------ primitives


def annualize(gross: Decimal, frequency: PayFrequency) -> Decimal | None:
    """Annualize a recurring gross amount. Returns None for irregular frequency."""
    periods = PERIODS_PER_YEAR.get(frequency)
    if periods is None:
        return None
    return (gross * periods).quantize(CENT, rounding=ROUND_HALF_UP)


def pct_difference(a: Decimal, b: Decimal) -> Decimal:
    """Absolute difference of ``a`` from reference ``b`` as a percentage of ``b``."""
    if b == 0:
        raise ValueError("reference value must be non-zero")
    return (abs(a - b) / b * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def variation_pct(values: list[Decimal]) -> Decimal:
    """(max - min) / mean, as a percentage."""
    mean = sum(values) / len(values)
    return ((max(values) - min(values)) / mean * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def period_length_days(start: date, end: date) -> int:
    return (end - start).days + 1


def normalize_person_name(name: str) -> str:
    stripped = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", stripped.casefold()).split())


def normalize_employer_name(name: str) -> str:
    tokens = normalize_person_name(name.replace("&", " and ")).split()
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _ref(ext: DocumentExtraction, field: str) -> InputRef:
    return InputRef(document_id=ext.document_id, field=field, value=to_iso(ext.value(field)))


# ----------------------------------------------------------------- calculation


def calculate_case(
    records: list[DocumentRecord],
    extractions: list[DocumentExtraction],
    as_of_date: date,
    settings: CalculationSettings | None = None,
) -> CaseCalculations:
    settings = settings or CalculationSettings()
    findings: list[Finding] = []
    by_id = {r.document_id: r for r in records}

    # --- document-level problems -------------------------------------------------
    for rec in records:
        if rec.load_error:
            findings.append(Finding(code="DOCUMENT_UNREADABLE", severity=Severity.CRITICAL, message=f"{rec.filename}: {rec.load_error}", document_ids=[rec.document_id]))
        if rec.injection_signals:
            findings.append(
                Finding(
                    code="PROMPT_INJECTION_SUSPECTED",
                    severity=Severity.CRITICAL,
                    message=f"{rec.filename} contains instruction-like text aimed at an AI system ({', '.join(rec.injection_signals)})",
                    document_ids=[rec.document_id],
                )
            )
    for ext in extractions:
        if ext.document_type is DocumentType.UNKNOWN and not by_id[ext.document_id].load_error:
            findings.append(Finding(code="UNCLASSIFIED_DOCUMENT", severity=Severity.CRITICAL, message=f"{by_id[ext.document_id].filename} is not a supported document type", document_ids=[ext.document_id]))
        if ext.missing_critical_fields:
            findings.append(
                Finding(
                    code="MISSING_CRITICAL_FIELD",
                    severity=Severity.CRITICAL,
                    message=f"{by_id[ext.document_id].filename} ({ext.document_type.value}) is missing: {', '.join(ext.missing_critical_fields)}",
                    document_ids=[ext.document_id],
                    fields=list(ext.missing_critical_fields),
                )
            )

    # --- duplicates --------------------------------------------------------------
    duplicate_ids = _find_duplicates(records, extractions, findings)

    usable = [e for e in extractions if e.document_id not in duplicate_ids and e.document_type is not DocumentType.UNKNOWN]
    stubs = [e for e in usable if e.document_type is DocumentType.PAY_STUB]
    letters = [e for e in usable if e.document_type is DocumentType.EMPLOYMENT_LETTER]
    banks = [e for e in usable if e.document_type is DocumentType.BANK_STATEMENT]

    # --- date plausibility -------------------------------------------------------
    for ext in stubs:
        _check_pay_stub_dates(ext, as_of_date, findings)
        gross, net, ytd = ext.gross_pay.value, ext.net_pay.value, ext.ytd_gross.value
        if gross is not None and net is not None and net > gross:
            findings.append(Finding(code="NET_EXCEEDS_GROSS", severity=Severity.WARNING, message=f"Net pay {net} exceeds gross pay {gross}", document_ids=[ext.document_id], fields=["net_pay", "gross_pay"]))
        if gross is not None and ytd is not None and ytd < gross:
            findings.append(Finding(code="YTD_LESS_THAN_GROSS", severity=Severity.WARNING, message=f"Year-to-date gross {ytd} is less than current gross {gross}", document_ids=[ext.document_id], fields=["ytd_gross", "gross_pay"]))
    for ext in letters:
        start = ext.employment_start_date.value
        if start is not None and start > as_of_date:
            findings.append(Finding(code="FUTURE_DATE", severity=Severity.WARNING, message=f"Employment start date {start} is after the as-of date {as_of_date}", document_ids=[ext.document_id], fields=["employment_start_date"]))
        if ext.employment_status.value is EmploymentStatus.TERMINATED:
            findings.append(Finding(code="EMPLOYMENT_NOT_ACTIVE", severity=Severity.CRITICAL, message="Employment letter states the employment has ended", document_ids=[ext.document_id], fields=["employment_status"]))

    # --- cross-document consistency ---------------------------------------------
    _check_consistency(usable, findings)

    # --- income ------------------------------------------------------------------
    annualized_by_stub: list[DerivedValue] = []
    for ext in stubs:
        gross, freq = ext.gross_pay.value, ext.pay_frequency.value
        if gross is None or freq is None:
            continue
        annualized_by_stub.append(
            DerivedValue(
                name="annualized_gross",
                value=annualize(gross, freq),
                method=f"gross_pay x {PERIODS_PER_YEAR[freq]} periods ({freq.value})" if freq in PERIODS_PER_YEAR else "not annualized: irregular frequency",
                inputs=[_ref(ext, "gross_pay"), _ref(ext, "pay_frequency")],
            )
        )

    gross_values = [e.gross_pay.value for e in stubs if e.gross_pay.value is not None]
    if len(gross_values) >= 2:
        variation = DerivedValue(
            name="gross_pay_variation_pct",
            value=variation_pct(gross_values),
            method="(max gross - min gross) / mean gross x 100 across non-duplicate pay stubs",
            inputs=[_ref(e, "gross_pay") for e in stubs if e.gross_pay.value is not None],
        )
    else:
        variation = DerivedValue(name="gross_pay_variation_pct", value=None, method="requires at least two pay stubs", assumptions=["Income variability cannot be assessed from a single pay stub"])

    frequencies = {e.pay_frequency.value for e in stubs if e.pay_frequency.value is not None}
    if len(frequencies) > 1:
        findings.append(Finding(code="PAY_FREQUENCY_CONFLICT", severity=Severity.WARNING, message=f"Pay stubs state different frequencies: {', '.join(sorted(f.value for f in frequencies))}", document_ids=[e.document_id for e in stubs], fields=["pay_frequency"]))

    income_pattern = "UNKNOWN"
    recurring = DerivedValue(name="recurring_annual_income", value=None, method="not determinable")
    complete_stubs = [e for e in stubs if e.gross_pay.value is not None and e.pay_frequency.value is not None]
    if complete_stubs and len(complete_stubs) == len(stubs):
        if PayFrequency.IRREGULAR in frequencies:
            income_pattern = "VARIABLE"
            findings.append(Finding(code="IRREGULAR_PAY_FREQUENCY", severity=Severity.WARNING, message="Pay frequency is irregular; income is not annualized", document_ids=[e.document_id for e in stubs if e.pay_frequency.value is PayFrequency.IRREGULAR], fields=["pay_frequency"]))
            recurring.method = "not annualized: irregular pay frequency"
        elif variation.value is not None and variation.value > settings.variable_income_threshold_pct:
            income_pattern = "VARIABLE"
            findings.append(Finding(code="VARIABLE_INCOME", severity=Severity.WARNING, message=f"Gross pay varies by {variation.value}% across pay stubs (threshold {settings.variable_income_threshold_pct}%); income is not annualized", document_ids=[e.document_id for e in stubs], fields=["gross_pay"]))
            recurring.method = "not annualized: variable gross pay"
        elif len(frequencies) == 1:
            income_pattern = "RECURRING"
            latest = max(complete_stubs, key=lambda e: (e.pay_date.value or e.pay_period_end.value or date.min))
            freq = latest.pay_frequency.value
            recurring = DerivedValue(
                name="recurring_annual_income",
                value=annualize(latest.gross_pay.value, freq),
                method=f"most recent pay stub gross_pay x {PERIODS_PER_YEAR[freq]} ({freq.value})",
                inputs=[_ref(latest, "gross_pay"), _ref(latest, "pay_frequency"), _ref(latest, "pay_date")],
                assumptions=[
                    "Gross pay is assumed to recur at the stated frequency",
                    f"Gross variation across stubs within {settings.variable_income_threshold_pct}% (or a single stub)",
                ],
            )
            if len(complete_stubs) == 1:
                recurring.assumptions.append("Only one pay stub available; variability not assessed")

    # --- salary comparison --------------------------------------------------------
    salary_letters = [e for e in letters if e.base_annual_salary.value is not None]
    salaries = {e.base_annual_salary.value for e in salary_letters}
    if len(salaries) > 1:
        findings.append(Finding(code="CONFLICTING_SALARY_DOCUMENTS", severity=Severity.WARNING, message=f"Employment letters state different base salaries: {', '.join(str(s) for s in sorted(salaries))}", document_ids=[e.document_id for e in salary_letters], fields=["base_annual_salary"]))
    documented = DerivedValue(name="documented_annual_salary", value=None, method="no explicitly documented base salary")
    if len(salaries) == 1:
        documented = DerivedValue(name="documented_annual_salary", value=salary_letters[0].base_annual_salary.value, method="base_annual_salary from employment letter", inputs=[_ref(e, "base_annual_salary") for e in salary_letters])
    variance = DerivedValue(name="salary_variance_pct", value=None, method="requires one documented salary and a recurring annualized income")
    if documented.value is not None and recurring.value is not None:
        variance = DerivedValue(
            name="salary_variance_pct",
            value=pct_difference(recurring.value, documented.value),
            method="|recurring_annual_income - documented_annual_salary| / documented_annual_salary x 100",
            inputs=recurring.inputs + documented.inputs,
        )

    # --- recency --------------------------------------------------------------------
    dated = [(e.pay_date.value or e.pay_period_end.value, e) for e in stubs if (e.pay_date.value or e.pay_period_end.value)]
    age = DerivedValue(name="latest_pay_stub_age_days", value=None, method="no dated pay stub")
    if dated:
        latest_date, latest_stub = max(dated, key=lambda item: item[0])
        field = "pay_date" if latest_stub.pay_date.value else "pay_period_end"
        age = DerivedValue(name="latest_pay_stub_age_days", value=Decimal((as_of_date - latest_date).days), method=f"as_of_date - latest pay stub {field}", inputs=[_ref(latest_stub, field)], assumptions=[f"as_of_date = {as_of_date.isoformat()}"])

    # --- bank corroboration ---------------------------------------------------------
    matched = DerivedValue(name="matched_payroll_deposits", value=None, method="requires a bank statement and a pay stub net pay")
    nets = [(e, e.net_pay.value) for e in stubs if e.net_pay.value is not None]
    if banks and nets:
        deposits = [d for b in banks for d in b.deposits]
        hits = [d for d in deposits if any(abs(d.amount - n) <= CENT for _, n in nets)]
        matched = DerivedValue(
            name="matched_payroll_deposits",
            value=Decimal(len(hits)),
            method="count of payroll deposits within $0.01 of a pay stub net pay",
            inputs=[_ref(e, "net_pay") for e, _ in nets] + [InputRef(document_id=d.source_document_id, field="deposits", value=f"{d.date.isoformat()} {d.amount}") for d in deposits],
        )
        if not hits:
            findings.append(Finding(code="DEPOSITS_DO_NOT_CORROBORATE", severity=Severity.WARNING, message="No payroll deposit on the bank statement matches a pay stub net pay", document_ids=[b.document_id for b in banks] + [e.document_id for e, _ in nets], fields=["deposits", "net_pay"]))

    return CaseCalculations(
        annualized_by_pay_stub=annualized_by_stub,
        recurring_annual_income=recurring,
        documented_annual_salary=documented,
        salary_variance_pct=variance,
        gross_pay_variation_pct=variation,
        latest_pay_stub_age_days=age,
        matched_payroll_deposits=matched,
        income_pattern=income_pattern,
        findings=findings,
    )


def _find_duplicates(records: list[DocumentRecord], extractions: list[DocumentExtraction], findings: list[Finding]) -> set[str]:
    """Exact (file hash), textual (normalized text hash) and semantic (same pay stub facts)."""
    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for rec in records:
        for key, kind in ((rec.sha256, "identical file"), (rec.text_sha256, "identical text")):
            if key is None:
                continue
            if key in seen and seen[key] != rec.document_id:
                duplicates.add(rec.document_id)
                findings.append(Finding(code="DUPLICATE_DOCUMENT", severity=Severity.WARNING, message=f"{rec.filename} duplicates {seen[key]} ({kind})", document_ids=[seen[key], rec.document_id]))
                break
            seen.setdefault(key, rec.document_id)
    signatures: dict[tuple, str] = {}
    for ext in extractions:
        if ext.document_type is not DocumentType.PAY_STUB or ext.document_id in duplicates:
            continue
        sig = (
            normalize_employer_name(ext.employer_name.value or ""),
            ext.pay_period_start.value,
            ext.pay_period_end.value,
            ext.pay_date.value,
            ext.gross_pay.value,
        )
        if all(v not in (None, "") for v in sig):
            if sig in signatures:
                duplicates.add(ext.document_id)
                findings.append(Finding(code="DUPLICATE_DOCUMENT", severity=Severity.WARNING, message=f"Pay stub {ext.document_id} reports the same employer, period, pay date and gross as {signatures[sig]}", document_ids=[signatures[sig], ext.document_id]))
            else:
                signatures[sig] = ext.document_id
    return duplicates


def _check_pay_stub_dates(ext: DocumentExtraction, as_of: date, findings: list[Finding]) -> None:
    start, end, paid, freq = ext.pay_period_start.value, ext.pay_period_end.value, ext.pay_date.value, ext.pay_frequency.value
    doc = [ext.document_id]
    for name, value in (("pay_period_start", start), ("pay_period_end", end), ("pay_date", paid)):
        if value is not None and value > as_of:
            findings.append(Finding(code="FUTURE_DATE", severity=Severity.WARNING, message=f"{name} {value} is after the as-of date {as_of}", document_ids=doc, fields=[name]))
    if start is not None and end is not None:
        if end < start:
            findings.append(Finding(code="INVALID_PAY_PERIOD", severity=Severity.WARNING, message=f"Pay period ends ({end}) before it starts ({start})", document_ids=doc, fields=["pay_period_start", "pay_period_end"]))
            return
        if freq in PERIOD_LENGTH_DAYS:
            low, high = PERIOD_LENGTH_DAYS[freq]
            length = period_length_days(start, end)
            if not low <= length <= high:
                findings.append(Finding(code="PAY_PERIOD_FREQUENCY_MISMATCH", severity=Severity.WARNING, message=f"Pay period is {length} days, inconsistent with {freq.value} frequency (expected {low}-{high})", document_ids=doc, fields=["pay_period_start", "pay_period_end", "pay_frequency"]))
    if end is not None and paid is not None:
        delta = (paid - end).days
        if delta < -PAY_DATE_MAX_DAYS_BEFORE_END or delta > PAY_DATE_MAX_DAYS_AFTER_END:
            findings.append(Finding(code="IMPLAUSIBLE_PAY_DATE", severity=Severity.WARNING, message=f"Pay date {paid} is {delta} days from period end {end} (allowed -{PAY_DATE_MAX_DAYS_BEFORE_END} to +{PAY_DATE_MAX_DAYS_AFTER_END})", document_ids=doc, fields=["pay_date", "pay_period_end"]))


def _check_consistency(usable: list[DocumentExtraction], findings: list[Finding]) -> None:
    def distinct(field: str, normalizer) -> dict[str, list[DocumentExtraction]]:  # noqa: ANN001
        groups: dict[str, list[DocumentExtraction]] = {}
        for ext in usable:
            value = ext.value(field)
            if value:
                groups.setdefault(normalizer(value), []).append(ext)
        return groups

    names = distinct("employee_name", normalize_person_name)
    if len(names) > 1:
        shown = sorted({e.employee_name.value for g in names.values() for e in g})
        findings.append(Finding(code="EMPLOYEE_NAME_MISMATCH", severity=Severity.CRITICAL, message=f"Documents name different people: {', '.join(shown)}", document_ids=[e.document_id for g in names.values() for e in g], fields=["employee_name"]))
    employers = distinct("employer_name", normalize_employer_name)
    if len(employers) > 1:
        shown = sorted({e.employer_name.value for g in employers.values() for e in g})
        findings.append(Finding(code="EMPLOYER_MISMATCH", severity=Severity.CRITICAL, message=f"Documents name different employers: {', '.join(shown)}", document_ids=[e.document_id for g in employers.values() for e in g], fields=["employer_name"]))
    currencies = distinct("currency", str)
    if len(currencies) > 1:
        findings.append(Finding(code="CURRENCY_MISMATCH", severity=Severity.WARNING, message=f"Documents use different currencies: {', '.join(sorted(currencies))}", document_ids=[e.document_id for g in currencies.values() for e in g], fields=["currency"]))
