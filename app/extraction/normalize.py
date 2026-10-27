"""Deterministic normalization, grounding and validation of raw extractions.

Every provider (LLM or mock) goes through this module, so the canonical output obeys
the same rules regardless of where raw values came from:

* money -> Decimal quantized to cents; dates -> ISO; enums -> canonical values
* ambiguous numeric dates (e.g. 04/05/2026) -> null with a warning, never a guess
* every value must be supported by evidence that actually occurs in the source text;
  ungrounded values are discarded (the main hallucination control)
* fields that do not apply to the document type are dropped
"""

import re
import unicodedata
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from app.models.common import (
    CRITICAL_FIELDS_BY_TYPE,
    DATE_FIELDS,
    FIELDS_BY_TYPE,
    MONEY_FIELDS,
    SCALAR_FIELDS,
    TEXT_FIELDS,
    DocumentType,
    EmploymentStatus,
    ExtractionStatus,
    PayFrequency,
    Severity,
)
from app.models.extraction import (
    BankDeposit,
    DocumentExtraction,
    ExtractedField,
    RawExtraction,
    RawField,
    ValidationIssue,
)

CENT = Decimal("0.01")
SUPPORTED_CURRENCIES = {"CAD", "USD", "EUR", "GBP"}

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}


class NormalizationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------ value parsers


def parse_money(raw: str) -> Decimal:
    cleaned = raw.strip().upper()
    for token in ("CAD", "USD", "EUR", "GBP", "C$", "US$", "$", "€", "£"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        raise NormalizationError("INVALID_MONEY", f"Cannot parse money value '{raw}'")
    try:
        value = Decimal(cleaned)
    except InvalidOperation as exc:
        raise NormalizationError("INVALID_MONEY", f"Cannot parse money value '{raw}'") from exc
    value = -value if negative else value
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def parse_date(raw: str) -> date:
    s = raw.strip().rstrip(".").replace(",", " ")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = " ".join(s.split())
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _mkdate(int(m[1]), int(m[2]), int(m[3]), raw)
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", s)
    if m:
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        if a <= 12 and b <= 12 and a != b:
            raise NormalizationError("AMBIGUOUS_DATE", f"Date '{raw}' is ambiguous (day/month order unknown)")
        if a > 12:
            return _mkdate(year, b, a, raw)  # DD/MM/YYYY
        return _mkdate(year, a, b, raw)  # MM/DD/YYYY (or a == b)
    m = re.fullmatch(r"([A-Za-z]+)\.? (\d{1,2}) (\d{4})", s)
    if m and m[1].lower() in _MONTHS:
        return _mkdate(int(m[3]), _MONTHS[m[1].lower()], int(m[2]), raw)
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+)\.? (\d{4})", s)
    if m and m[2].lower() in _MONTHS:
        return _mkdate(int(m[3]), _MONTHS[m[2].lower()], int(m[1]), raw)
    raise NormalizationError("INVALID_DATE", f"Cannot parse date '{raw}'")


def _mkdate(year: int, month: int, day: int, raw: str) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise NormalizationError("INVALID_DATE", f"'{raw}' is not a valid calendar date") from exc


_FREQUENCY_ALIASES: list[tuple[str, PayFrequency]] = [
    (r"semi[\s-]?monthly|twice (a|per) month|24 pay periods", PayFrequency.SEMIMONTHLY),
    (r"bi[\s-]?weekly|every (two|2) weeks|fortnightly|26 pay periods", PayFrequency.BIWEEKLY),
    (r"weekly|every week|52 pay periods", PayFrequency.WEEKLY),
    (r"monthly|once a month|12 pay periods", PayFrequency.MONTHLY),
    (r"irregular|variable|per assignment|per project|ad hoc|on demand", PayFrequency.IRREGULAR),
]


def parse_frequency(raw: str) -> PayFrequency:
    s = raw.strip().lower()
    for pattern, freq in _FREQUENCY_ALIASES:
        if re.search(pattern, s):
            return freq
    raise NormalizationError("INVALID_FREQUENCY", f"Unrecognized pay frequency '{raw}'")


def parse_status(raw: str) -> EmploymentStatus:
    s = raw.strip().lower()
    if re.search(r"terminat|ended|former|no longer|resign", s):
        return EmploymentStatus.TERMINATED
    if re.search(r"part[\s_-]?time", s):
        return EmploymentStatus.PART_TIME
    if re.search(r"contract|temporary|freelance|independent", s):
        return EmploymentStatus.CONTRACT
    if re.search(r"full[\s_-]?time|permanent", s):
        return EmploymentStatus.FULL_TIME
    raise NormalizationError("INVALID_STATUS", f"Unrecognized employment status '{raw}'")


def parse_currency(raw: str) -> str:
    code = raw.strip().upper()
    if code not in SUPPORTED_CURRENCIES:
        raise NormalizationError("INVALID_CURRENCY", f"Unsupported or ambiguous currency '{raw}'")
    return code


def parse_text(raw: str) -> str:
    value = " ".join(raw.split()).strip(" ,;:")
    if not value:
        raise NormalizationError("EMPTY_VALUE", "Empty text value")
    return value


def _parser_for(field: str):  # noqa: ANN202
    if field in MONEY_FIELDS:
        return parse_money
    if field in DATE_FIELDS:
        return parse_date
    return {
        "pay_frequency": parse_frequency,
        "employment_status": parse_status,
        "currency": parse_currency,
    }.get(field, parse_text)


# ------------------------------------------------------------------- grounding


def _canon(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return " ".join(text.split()).casefold()


def evidence_in_source(evidence: str, source_canon: str) -> bool:
    return bool(evidence.strip()) and _canon(evidence) in source_canon


def value_supported_by_evidence(field: str, raw_value: str, evidence: str) -> bool:
    if field in MONEY_FIELDS:
        digits = re.sub(r"\D", "", raw_value.split(".")[0])
        return bool(digits) and digits in re.sub(r"\D", "", evidence)
    if field in TEXT_FIELDS:
        return _canon(raw_value) in _canon(evidence)
    return True  # dates and enums are legitimately reformatted (e.g. "every two weeks" -> BIWEEKLY)


# ---------------------------------------------------------------- normalization


def _normalize_field(field: str, raw: RawField, document_id: str, source_canon: str, issues: list[ValidationIssue]) -> ExtractedField:
    out = ExtractedField(source_document_id=document_id, model_confidence=raw.confidence)
    if raw.value is None or not raw.value.strip():
        return out

    def reject(code: str, message: str, severity: Severity = Severity.WARNING) -> ExtractedField:
        out.warnings.append(f"{code}: {message}")
        issues.append(ValidationIssue(code=code, message=message, field=field, document_id=document_id, severity=severity))
        return out

    if not raw.evidence or not evidence_in_source(raw.evidence, source_canon):
        return reject("EVIDENCE_NOT_GROUNDED", f"Discarded '{raw.value}': supporting evidence not found verbatim in document")
    if not value_supported_by_evidence(field, raw.value, raw.evidence):
        return reject("VALUE_NOT_IN_EVIDENCE", f"Discarded '{raw.value}': value does not appear in its evidence")
    try:
        out.value = _parser_for(field)(raw.value)
    except NormalizationError as exc:
        return reject(exc.code, str(exc))
    out.evidence = raw.evidence.strip()
    if field in MONEY_FIELDS and out.value is not None and out.value <= 0:
        out.value = None
        return reject("NON_POSITIVE_AMOUNT", f"Discarded non-positive amount '{raw.value}'")
    return out


def normalize_extraction(raw: RawExtraction, document_id: str, source_text: str) -> DocumentExtraction:
    doc_type = DocumentType(raw.document_type)
    applicable = set(FIELDS_BY_TYPE[doc_type])
    source_canon = _canon(source_text)
    issues: list[ValidationIssue] = []
    fields: dict[str, ExtractedField] = {}

    for name in SCALAR_FIELDS:
        raw_field: RawField = getattr(raw, name)
        if name not in applicable:
            if raw_field.value is not None:
                issues.append(
                    ValidationIssue(
                        code="FIELD_NOT_APPLICABLE",
                        message=f"Ignored '{name}' because it does not apply to a {doc_type.value}",
                        field=name,
                        document_id=document_id,
                        severity=Severity.INFO,
                    )
                )
            fields[name] = ExtractedField(source_document_id=document_id)
            continue
        fields[name] = _normalize_field(name, raw_field, document_id, source_canon, issues)

    deposits: list[BankDeposit] = []
    if "deposits" in applicable:
        for i, dep in enumerate(raw.deposits):
            try:
                if not dep.evidence or not evidence_in_source(dep.evidence, source_canon):
                    raise NormalizationError("EVIDENCE_NOT_GROUNDED", "deposit evidence not found in document")
                if dep.date is None or dep.amount is None:
                    raise NormalizationError("INCOMPLETE_DEPOSIT", "deposit is missing a date or amount")
                if not value_supported_by_evidence("gross_pay", dep.amount, dep.evidence):
                    raise NormalizationError("VALUE_NOT_IN_EVIDENCE", "deposit amount not in its evidence")
                amount = parse_money(dep.amount)
                if amount <= 0:
                    raise NormalizationError("NON_POSITIVE_AMOUNT", "deposits must be positive")
                deposits.append(
                    BankDeposit(
                        date=parse_date(dep.date),
                        amount=amount,
                        description=dep.description,
                        evidence=dep.evidence.strip(),
                        source_document_id=document_id,
                    )
                )
            except NormalizationError as exc:
                issues.append(
                    ValidationIssue(code=exc.code, message=f"Deposit #{i + 1} discarded: {exc}", field="deposits", document_id=document_id)
                )

    missing = [f for f in CRITICAL_FIELDS_BY_TYPE[doc_type] if fields[f].value is None]
    for f in missing:
        issues.append(
            ValidationIssue(
                code="MISSING_CRITICAL_FIELD",
                message=f"Critical field '{f}' is missing or could not be validated",
                field=f,
                document_id=document_id,
                severity=Severity.CRITICAL,
            )
        )
    if doc_type is DocumentType.UNKNOWN:
        issues.append(
            ValidationIssue(
                code="UNCLASSIFIED_DOCUMENT",
                message="Document could not be classified as a supported type",
                document_id=document_id,
                severity=Severity.CRITICAL,
            )
        )
    status = ExtractionStatus.SUCCESS if not missing and doc_type is not DocumentType.UNKNOWN else ExtractionStatus.PARTIAL

    return DocumentExtraction(
        document_id=document_id,
        document_type=doc_type,
        deposits=deposits,
        status=status,
        issues=issues,
        missing_critical_fields=missing,
        model_flagged_suspicious=raw.suspicious_instructions_found,
        **fields,  # type: ignore[arg-type]
    )


def to_iso(value: object) -> str | None:
    """Canonical string form used for comparisons and audit payloads."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
