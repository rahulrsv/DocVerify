from datetime import date
from decimal import Decimal

import pytest

from app.calculations.income import (
    annualize,
    calculate_case,
    normalize_employer_name,
    normalize_person_name,
    pct_difference,
    period_length_days,
    variation_pct,
)
from app.models.case import DocumentRecord
from app.models.common import DocumentType, EmploymentStatus, PayFrequency
from app.models.extraction import BankDeposit
from tests.conftest import AS_OF, letter, make_extraction, record, stub


@pytest.mark.parametrize(
    ("gross", "freq", "expected"),
    [
        ("1250.00", PayFrequency.WEEKLY, "65000.00"),
        ("3076.92", PayFrequency.BIWEEKLY, "79999.92"),
        ("4166.67", PayFrequency.SEMIMONTHLY, "100000.08"),
        ("7500.00", PayFrequency.MONTHLY, "90000.00"),
    ],
)
def test_annualize_supported_frequencies(gross, freq, expected):
    assert annualize(Decimal(gross), freq) == Decimal(expected)


def test_annualize_refuses_irregular_income():
    assert annualize(Decimal("4120.50"), PayFrequency.IRREGULAR) is None


def test_pct_difference_and_variation():
    assert pct_difference(Decimal("79999.92"), Decimal("82000")) == Decimal("2.44")
    assert variation_pct([Decimal("2100"), Decimal("3450")]) == Decimal("48.65")
    with pytest.raises(ValueError):
        pct_difference(Decimal("1"), Decimal("0"))


def test_period_length_is_inclusive():
    assert period_length_days(date(2026, 9, 14), date(2026, 9, 27)) == 14


def test_name_normalization():
    assert normalize_person_name("  Lucas  Fernández ") == "lucas fernandez"
    assert normalize_employer_name("Aurora Dental Group Inc.") == normalize_employer_name("Aurora Dental Group")
    assert normalize_employer_name("Harbour Logistics Ltd.") != normalize_employer_name("Pinecrest Logistics Ltd.")


def _calc(extractions, records=None, as_of=AS_OF):
    records = records or [record(e.document_id) for e in extractions]
    return calculate_case(records, extractions, as_of)


def codes(calc):
    return {f.code for f in calc.findings}


def test_clean_case_derives_recurring_income_with_provenance():
    calc = _calc([stub(), letter()])
    assert calc.income_pattern == "RECURRING"
    assert calc.recurring_annual_income.value == Decimal("79999.92")
    assert {i.field for i in calc.recurring_annual_income.inputs} == {"gross_pay", "pay_frequency", "pay_date"}
    assert calc.salary_variance_pct.value == Decimal("0.00")
    assert calc.latest_pay_stub_age_days.value == Decimal(13)
    assert codes(calc) == set()


def test_salary_conflict_variance():
    calc = _calc([stub(), letter(salary="120000.00")])
    assert calc.salary_variance_pct.value == Decimal("33.33")


def test_conflicting_salary_letters():
    calc = _calc([stub(), letter("D2", salary="70000.00"), letter("D3", salary="85000.00")])
    assert "CONFLICTING_SALARY_DOCUMENTS" in codes(calc)
    assert calc.documented_annual_salary.value is None


def test_variable_income_is_not_annualized():
    calc = _calc([stub("D1", gross="2100.00", start=date(2026, 8, 31), end=date(2026, 9, 13), paid=date(2026, 9, 18)), stub("D2", gross="3450.00"), letter("D3")])
    assert calc.income_pattern == "VARIABLE"
    assert calc.recurring_annual_income.value is None
    assert "VARIABLE_INCOME" in codes(calc)


def test_irregular_frequency_is_not_annualized():
    calc = _calc([stub(freq=PayFrequency.IRREGULAR), letter()])
    assert calc.income_pattern == "VARIABLE"
    assert calc.recurring_annual_income.value is None
    assert "IRREGULAR_PAY_FREQUENCY" in codes(calc)


def test_missing_gross_leaves_income_unknown():
    s = stub(gross=None, missing=["gross_pay"])
    calc = _calc([s, letter()])
    assert calc.income_pattern == "UNKNOWN"
    assert calc.recurring_annual_income.value is None
    assert "MISSING_CRITICAL_FIELD" in codes(calc)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (dict(start=date(2026, 9, 27), end=date(2026, 9, 14)), "INVALID_PAY_PERIOD"),
        (dict(freq=PayFrequency.WEEKLY, start=date(2026, 9, 1), end=date(2026, 9, 30)), "PAY_PERIOD_FREQUENCY_MISMATCH"),
        (dict(start=date(2026, 11, 30), end=date(2026, 12, 13), paid=date(2026, 12, 18)), "FUTURE_DATE"),
        (dict(paid=date(2026, 10, 30)), "IMPLAUSIBLE_PAY_DATE"),
        (dict(ytd_gross=Decimal("100.00")), "YTD_LESS_THAN_GROSS"),
        (dict(net_pay=Decimal("9999.00")), "NET_EXCEEDS_GROSS"),
    ],
)
def test_pay_stub_plausibility_findings(kwargs, code):
    assert code in codes(_calc([stub(**kwargs), letter()]))


def test_future_employment_start_and_terminated_status():
    calc = _calc([stub(), letter(employment_start_date=date(2027, 1, 10), employment_status=EmploymentStatus.TERMINATED)])
    assert {"FUTURE_DATE", "EMPLOYMENT_NOT_ACTIVE"} <= codes(calc)


def test_identity_employer_and_currency_mismatch():
    calc = _calc([stub(), letter(employee_name="Lucia Fernandes", employer_name="Pinecrest Logistics Ltd.", currency="USD")])
    assert {"EMPLOYEE_NAME_MISMATCH", "EMPLOYER_MISMATCH", "CURRENCY_MISMATCH"} <= codes(calc)


def test_duplicate_by_file_hash_excluded_from_income():
    recs = [record("D1", sha="same"), record("D2", sha="same"), record("D3")]
    calc = _calc([stub("D1"), stub("D2"), letter("D3")], recs)
    assert "DUPLICATE_DOCUMENT" in codes(calc)
    assert calc.gross_pay_variation_pct.value is None  # duplicate not counted as a second stub


def test_duplicate_by_pay_stub_signature():
    calc = _calc([stub("D1"), stub("D2"), letter("D3")])
    assert [f.code for f in calc.findings].count("DUPLICATE_DOCUMENT") == 1


def test_frequency_conflict():
    calc = _calc([stub("D1"), stub("D2", freq=PayFrequency.SEMIMONTHLY, start=date(2026, 9, 16), end=date(2026, 9, 30), paid=date(2026, 9, 30)), letter("D3")])
    assert "PAY_FREQUENCY_CONFLICT" in codes(calc)
    assert calc.recurring_annual_income.value is None


def test_deposit_corroboration():
    bank = make_extraction(
        "D3", DocumentType.BANK_STATEMENT, employee_name="Jordan Avery", currency="CAD",
        deposits=[BankDeposit(date=date(2026, 10, 2), amount=Decimal("2284.11"), evidence="x", source_document_id="D3")],
    )
    assert _calc([stub(), letter(), bank]).matched_payroll_deposits.value == Decimal(1)
    bad = bank.model_copy(update={"deposits": [BankDeposit(date=date(2026, 10, 2), amount=Decimal("1200.00"), evidence="x", source_document_id="D3")]})
    assert "DEPOSITS_DO_NOT_CORROBORATE" in codes(_calc([stub(), letter(), bad]))


def test_unreadable_and_injection_findings():
    recs = [DocumentRecord(document_id="D1", filename="a.pdf", media_type="application/pdf", sha256="a", load_error="NO_TEXT_LAYER"), record("D2", injection_signals=["IGNORE_INSTRUCTIONS"])]
    calc = _calc([make_extraction("D1", DocumentType.UNKNOWN), letter("D2")], recs)
    assert {"DOCUMENT_UNREADABLE", "PROMPT_INJECTION_SUSPECTED"} <= codes(calc)


def test_calculations_are_deterministic():
    a = _calc([stub(), letter()])
    b = _calc([stub(), letter()])
    assert a.model_dump() == b.model_dump()
