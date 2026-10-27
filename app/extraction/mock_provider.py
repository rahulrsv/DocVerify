"""Deterministic rule-based extraction provider ("mock").

This is NOT an LLM and does not simulate one. It is a regex/heuristic extractor that
lets the whole pipeline, test suite and evaluation run offline and reproducibly. Its
evaluation metrics measure this heuristic baseline only and are labeled as such.
It reports no confidence because rules have no meaningful confidence estimate.
"""

import re
import time

from app.documents.classifier import classify
from app.extraction.base import ProviderResult
from app.models.common import DocumentType
from app.models.extraction import RawDeposit, RawExtraction, RawField

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{4}}/\d{{2}}/\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|{_MONTH} \d{{1,2}},? \d{{4}}|\d{{1,2}} {_MONTH} \d{{4}})"
MONEY = r"\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
_COMPANY_SUFFIX = r"(?:Inc|Ltd|LLP|LLC|Corp|Co|Group|Partners|Limited|Corporation)\b\.?"
_FLAGS = re.IGNORECASE | re.MULTILINE


def _field(value: str | None, evidence: str | None) -> RawField:
    if value is None:
        return RawField.empty()
    return RawField(value=value.strip(), evidence=(evidence or value).strip(), confidence=None)


def _line_of(text: str, match: re.Match[str]) -> str:
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    return text[start : end if end != -1 else len(text)].strip()


def _labeled(text: str, labels: str, value_pattern: str, to_eol: bool = False) -> RawField:
    """First line that starts with one of ``labels`` followed by ``value_pattern``."""
    tail = r"[ \t]*$" if to_eol else ""
    m = re.search(rf"^[ \t]*(?:{labels})[ \t]*[:\-]?[ \t]*({value_pattern}){tail}", text, _FLAGS)
    return _field(m.group(1), _line_of(text, m)) if m else RawField.empty()


def _currency(text: str) -> RawField:
    m = re.search(r"\b(CAD|USD|EUR|GBP)\b", text)
    return _field(m.group(1), _line_of(text, m)) if m else RawField.empty()


class RuleBasedExtractionProvider:
    name = "mock"
    model = "rules-extractor-1.0"
    prompt_version = "n/a (rule-based)"

    def extract(self, text: str, document_id: str) -> ProviderResult:
        started = time.perf_counter()
        doc_type, _scores = classify(text)
        raw = RawExtraction.blank(doc_type.value)
        if doc_type is DocumentType.PAY_STUB:
            raw = self._pay_stub(text)
        elif doc_type is DocumentType.EMPLOYMENT_LETTER:
            raw = self._letter(text)
        elif doc_type is DocumentType.BANK_STATEMENT:
            raw = self._bank(text)
        return ProviderResult(raw=raw, attempts=1, latency_ms=int((time.perf_counter() - started) * 1000))

    # ------------------------------------------------------------------ pay stub

    def _pay_stub(self, text: str) -> RawExtraction:
        raw = RawExtraction.blank(DocumentType.PAY_STUB.value)
        raw.employee_name = _labeled(text, r"employee(?: name)?|pay to|payee", r"[A-Z][^\n:]*?[A-Za-z]", to_eol=True)
        raw.employer_name = _labeled(text, r"employer|company", r"[A-Z][^\n]*?[A-Za-z.]", to_eol=True)
        period = re.search(rf"^[ \t]*(?:pay period|period)[ \t]*:?[ \t]*({DATE})\s*(?:to|-|–|through)\s*({DATE})", text, _FLAGS)
        if period:
            line = _line_of(text, period)
            raw.pay_period_start = _field(period.group(1), line)
            raw.pay_period_end = _field(period.group(2), line)
        raw.pay_date = _labeled(text, r"pay date|paid on|payment date|deposit date", DATE)
        raw.pay_frequency = _labeled(text, r"pay frequency|frequency|pay schedule", r"[A-Za-z][A-Za-z \-]*[A-Za-z]", to_eol=True)

        has_ytd_column = re.search(r"year-to-date|\bytd\b", text, re.IGNORECASE) is not None
        gross_row = re.search(rf"^[ \t]*(?:gross pay|gross earnings|total gross)[ \t]*:?[ \t]*({MONEY})(?:[ \t]+({MONEY}))?[ \t]*$", text, _FLAGS)
        if gross_row:
            line = _line_of(text, gross_row)
            raw.gross_pay = _field(gross_row.group(1), line)
            if gross_row.group(2) and has_ytd_column:
                raw.ytd_gross = _field(gross_row.group(2), line)
        net_row = re.search(rf"^[ \t]*net pay[ \t]*:?[ \t]*({MONEY})", text, _FLAGS)
        if net_row:
            raw.net_pay = _field(net_row.group(1), _line_of(text, net_row))
        if raw.ytd_gross.value is None:
            raw.ytd_gross = _labeled(text, r"ytd gross(?: earnings)?|year-to-date gross(?: earnings)?", MONEY)
        raw.currency = _labeled(text, r"currency|all amounts in", r"[A-Z]{3}")
        if raw.currency.value is None:
            raw.currency = _currency(text)
        return raw

    # -------------------------------------------------------------------- letter

    def _letter(self, text: str) -> RawExtraction:
        raw = RawExtraction.blank(DocumentType.EMPLOYMENT_LETTER.value)
        # Letters are prose and may wrap mid-sentence; search a whitespace-flattened copy.
        # Evidence taken from it still passes grounding, which compares normalized whitespace.
        flat = " ".join(text.split())
        name = re.search(r"employment verification\s*[-–—:]\s*([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)", flat)
        if not name:
            name = re.search(r"confirms? that ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+) (?:has been|is|was)", flat)
        if name:
            raw.employee_name = _field(name.group(1), name.group(0))

        employer = re.search(rf"employed (?:by|with|at) ([A-Z][A-Za-z&'\- ]*?{_COMPANY_SUFFIX})", flat)
        if employer:
            raw.employer_name = _field(employer.group(1), employer.group(0))
        else:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            if re.search(rf"{_COMPANY_SUFFIX}$", first_line):
                raw.employer_name = _field(first_line, first_line)

        start = re.search(rf"(?:since|from|start date (?:of|was|is)|commenced employment on|hired on)\s+({DATE})", flat, re.IGNORECASE)
        if start:
            raw.employment_start_date = _field(start.group(1), start.group(0))

        status = re.search(r"employment (?:ended|was terminated|terminated) on|no longer employed", flat, re.IGNORECASE) or re.search(
            r"\b(full-time|full time|part-time|part time|contract|temporary)\b", flat, re.IGNORECASE
        )
        if status:
            raw.employment_status = _field(status.group(0), status.group(0))

        salary = re.search(rf"base annual salary (?:is|of)\s*({MONEY})(?:\s*(CAD|USD|EUR|GBP))?", flat, re.IGNORECASE)
        if salary:
            raw.base_annual_salary = _field(salary.group(1), salary.group(0))
            if salary.group(2):
                raw.currency = _field(salary.group(2), salary.group(0))
        if raw.currency.value is None:
            raw.currency = _currency(text)
        return raw

    # ---------------------------------------------------------------------- bank

    def _bank(self, text: str) -> RawExtraction:
        raw = RawExtraction.blank(DocumentType.BANK_STATEMENT.value)
        raw.employee_name = _labeled(text, r"account holder", r"[A-Z][^\n]*?[A-Za-z]", to_eol=True)
        raw.currency = _labeled(text, r"currency", r"[A-Z]{3}")
        if raw.currency.value is None:
            raw.currency = _currency(text)
        for m in re.finditer(rf"^[ \t]*({DATE})[ \t]+(.+?)[ \t]+(-?{MONEY})[ \t]*$", text, _FLAGS):
            description, amount = m.group(2).strip(), m.group(3)
            if amount.startswith("-") or not re.search(r"payroll|direct dep|salary|pay dep", description, re.IGNORECASE):
                continue
            raw.deposits.append(RawDeposit(date=m.group(1), amount=amount, description=description, evidence=_line_of(text, m)))
        return raw
