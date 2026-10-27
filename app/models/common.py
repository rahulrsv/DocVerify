"""Shared enums and version constants."""

from enum import StrEnum

SCHEMA_VERSION = "extraction-schema-1.0"

DISCLAIMER = (
    "Synthetic engineering demonstration only. Outcomes reflect configured demo rules "
    "applied to synthetic documents and are not credit decisions or financial advice."
)


class DocumentType(StrEnum):
    PAY_STUB = "pay_stub"
    EMPLOYMENT_LETTER = "employment_letter"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


class PayFrequency(StrEnum):
    WEEKLY = "WEEKLY"
    BIWEEKLY = "BIWEEKLY"
    SEMIMONTHLY = "SEMIMONTHLY"
    MONTHLY = "MONTHLY"
    IRREGULAR = "IRREGULAR"


class EmploymentStatus(StrEnum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    TERMINATED = "TERMINATED"


class ExtractionStatus(StrEnum):
    SUCCESS = "SUCCESS"  # all critical fields present and grounded
    PARTIAL = "PARTIAL"  # extraction ran but critical fields are missing
    FAILED = "FAILED"  # document could not be loaded or the provider failed


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Outcome(StrEnum):
    APPROVE = "APPROVE"
    REFER_FOR_REVIEW = "REFER_FOR_REVIEW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RuleOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Fields that are meaningful for each document type. Extracted values for fields
# outside this set are dropped during normalization so that, for example, a payroll
# deposit description on a bank statement is not mistaken for a documented employer.
FIELDS_BY_TYPE: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PAY_STUB: (
        "employee_name",
        "employer_name",
        "pay_period_start",
        "pay_period_end",
        "pay_date",
        "gross_pay",
        "net_pay",
        "pay_frequency",
        "ytd_gross",
        "currency",
    ),
    DocumentType.EMPLOYMENT_LETTER: (
        "employee_name",
        "employer_name",
        "employment_start_date",
        "employment_status",
        "base_annual_salary",
        "currency",
    ),
    DocumentType.BANK_STATEMENT: ("employee_name", "currency", "deposits"),
    DocumentType.UNKNOWN: (),
}

# Fields that must be present for a document of that type to count as usable evidence.
CRITICAL_FIELDS_BY_TYPE: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PAY_STUB: (
        "employee_name",
        "employer_name",
        "pay_period_start",
        "pay_period_end",
        "pay_date",
        "gross_pay",
        "pay_frequency",
    ),
    DocumentType.EMPLOYMENT_LETTER: (
        "employee_name",
        "employer_name",
        "employment_start_date",
        "employment_status",
    ),
    DocumentType.BANK_STATEMENT: ("employee_name",),
    DocumentType.UNKNOWN: (),
}

SCALAR_FIELDS: tuple[str, ...] = (
    "employee_name",
    "employer_name",
    "pay_period_start",
    "pay_period_end",
    "pay_date",
    "gross_pay",
    "net_pay",
    "pay_frequency",
    "ytd_gross",
    "employment_start_date",
    "employment_status",
    "base_annual_salary",
    "currency",
)

MONEY_FIELDS = frozenset({"gross_pay", "net_pay", "ytd_gross", "base_annual_salary"})
DATE_FIELDS = frozenset({"pay_period_start", "pay_period_end", "pay_date", "employment_start_date"})
ENUM_FIELDS = frozenset({"pay_frequency", "employment_status", "currency"})
TEXT_FIELDS = frozenset({"employee_name", "employer_name"})
