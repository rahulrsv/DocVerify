"""Case processing pipeline.

Document upload -> text extraction -> injection scan -> classification + structured
extraction (provider) -> normalization/grounding/validation -> deterministic
calculations -> policy evaluation -> audit record.
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

from app.audit.store import AuditStore
from app.calculations.income import calculate_case
from app.documents.loader import LoadedDocument, load_document
from app.documents.safety import scan_for_injection
from app.extraction.base import ExtractionProvider, ProviderError
from app.extraction.normalize import normalize_extraction
from app.models.case import CaseResult, ProcessingMetadata
from app.models.common import SCHEMA_VERSION, Severity
from app.models.extraction import DocumentExtraction, ValidationIssue
from app.policies.config import PolicyConfig
from app.policies.engine import decide, evaluate_rules
from app.policies.rules import PolicyContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentInput:
    filename: str
    content: bytes


class CaseProcessor:
    def __init__(
        self,
        provider: ExtractionProvider,
        policy: PolicyConfig,
        audit: AuditStore,
        *,
        max_pdf_pages: int = 20,
        max_text_chars: int = 50_000,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.audit = audit
        self._max_pages = max_pdf_pages
        self._max_chars = max_text_chars

    def fingerprint(self, documents: list[DocumentInput], as_of_date: date) -> str:
        """Stable identity of (inputs, extraction config, policy) used for idempotency."""
        material = {
            "documents": [hashlib.sha256(d.content).hexdigest() for d in documents],
            "as_of_date": as_of_date.isoformat(),
            "provider": self.provider.name,
            "model": self.provider.model,
            "prompt_version": self.provider.prompt_version,
            "schema_version": SCHEMA_VERSION,
            "policy_version": self.policy.policy_version,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def process_case(
        self,
        case_id: str,
        documents: list[DocumentInput],
        *,
        as_of_date: date | None = None,
        force: bool = False,
        source: str = "api",
        evaluation_run_id: str | None = None,
    ) -> CaseResult:
        started = time.perf_counter()
        as_of = as_of_date or datetime.now(UTC).date()
        fingerprint = self.fingerprint(documents, as_of)

        if not force:
            previous = self.audit.find_run_by_fingerprint(case_id, fingerprint)
            if previous is not None:
                self.audit.append_event(case_id, "PROCESSING_REPLAYED", {"input_fingerprint": fingerprint, "original_run_id": previous.run_id}, previous.run_id)
                return previous.model_copy(update={"idempotent_replay": True})

        run_id = f"run_{uuid.uuid4().hex[:16]}"
        errors: list[str] = []
        attempts: dict[str, int] = {}
        self.audit.append_event(
            case_id,
            "CASE_RECEIVED",
            {"document_count": len(documents), "filenames": [d.filename for d in documents], "as_of_date": as_of.isoformat(), "source": source, "evaluation_run_id": evaluation_run_id, "input_fingerprint": fingerprint},
            run_id,
        )

        loaded: list[LoadedDocument] = []
        extractions: list[DocumentExtraction] = []
        for index, doc in enumerate(documents):
            item = load_document(index, doc.filename, doc.content, max_pages=self._max_pages, max_chars=self._max_chars)
            loaded.append(item)
            rec = item.record
            if rec.load_error:
                errors.append(f"{rec.document_id}: {rec.load_error}")
                self.audit.append_event(case_id, "DOCUMENT_LOAD_FAILED", {"document_id": rec.document_id, "sha256": rec.sha256, "error": rec.load_error}, run_id)
                extractions.append(DocumentExtraction.failed(rec.document_id, ValidationIssue(code="DOCUMENT_UNREADABLE", message=rec.load_error, severity=Severity.CRITICAL, document_id=rec.document_id)))
                continue
            rec.injection_signals = scan_for_injection(item.text)
            self.audit.append_event(
                case_id,
                "DOCUMENT_LOADED",
                {"document_id": rec.document_id, "filename": rec.filename, "sha256": rec.sha256, "text_sha256": rec.text_sha256, "pages": rec.page_count, "chars": rec.char_count},
                run_id,
            )
            if rec.injection_signals:
                self.audit.append_event(case_id, "INJECTION_SIGNAL_DETECTED", {"document_id": rec.document_id, "signals": rec.injection_signals}, run_id)
            extractions.append(self._extract(case_id, run_id, item, attempts, errors))

        records = [item.record for item in loaded]
        calculations = calculate_case(records, extractions, as_of, self.policy.calculation_settings())
        self.audit.append_event(
            case_id,
            "CALCULATIONS_COMPLETED",
            {
                "recurring_annual_income": str(calculations.recurring_annual_income.value),
                "income_pattern": calculations.income_pattern,
                "findings": [f.code for f in calculations.findings],
            },
            run_id,
        )

        ctx = PolicyContext(records=records, extractions=extractions, calculations=calculations, as_of_date=as_of)
        rule_results = evaluate_rules(ctx, self.policy)
        decision = decide(rule_results, self.policy)
        self.audit.append_event(
            case_id,
            "POLICY_EVALUATED",
            {"policy_version": self.policy.policy_version, "rules": {r.rule_id: r.outcome.value for r in rule_results}},
            run_id,
        )

        result = CaseResult(
            run_id=run_id,
            case_id=case_id,
            processed_at=datetime.now(UTC),
            as_of_date=as_of,
            input_fingerprint=fingerprint,
            documents=records,
            extractions=extractions,
            calculations=calculations,
            rule_results=rule_results,
            decision=decision,
            metadata=ProcessingMetadata(
                provider=self.provider.name,
                model=self.provider.model,
                prompt_version=self.provider.prompt_version,
                schema_version=SCHEMA_VERSION,
                policy_version=self.policy.policy_version,
                extraction_attempts=attempts,
                errors=errors,
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
            source=source,
            evaluation_run_id=evaluation_run_id,
        )
        self.audit.save_run(result)
        self.audit.append_event(
            case_id,
            "DECISION_RECORDED",
            {"outcome": decision.outcome.value, "failed_rules": [r.rule_id for r in rule_results if r.outcome.value == "FAIL"], "duration_ms": result.duration_ms},
            run_id,
        )
        log.info(
            "case processed",
            extra={"ctx": {"case_id": case_id, "run_id": run_id, "outcome": decision.outcome.value, "duration_ms": result.duration_ms, "provider": self.provider.name}},
        )
        return result

    def _extract(self, case_id: str, run_id: str, item: LoadedDocument, attempts: dict[str, int], errors: list[str]) -> DocumentExtraction:
        doc_id = item.record.document_id
        try:
            provider_result = self.provider.extract(item.text, doc_id)
        except ProviderError as exc:
            attempts[doc_id] = exc.attempts
            errors.append(f"{doc_id}: extraction {exc.kind}: {exc}")
            self.audit.append_event(case_id, "EXTRACTION_FAILED", {"document_id": doc_id, "kind": exc.kind, "error": str(exc), "attempts": exc.attempts}, run_id)
            return DocumentExtraction.failed(doc_id, ValidationIssue(code=f"EXTRACTION_{exc.kind.upper()}", message=str(exc), severity=Severity.CRITICAL, document_id=doc_id))
        attempts[doc_id] = provider_result.attempts
        extraction = normalize_extraction(provider_result.raw, doc_id, item.text)
        self.audit.append_event(
            case_id,
            "EXTRACTION_COMPLETED",
            {
                "document_id": doc_id,
                "document_type": extraction.document_type.value,
                "status": extraction.status.value,
                "attempts": provider_result.attempts,
                "latency_ms": provider_result.latency_ms,
                "provider": self.provider.name,
                "model": self.provider.model,
                "prompt_version": self.provider.prompt_version,
                "issues": [i.code for i in extraction.issues],
                "model_flagged_suspicious": extraction.model_flagged_suspicious,
            },
            run_id,
        )
        return extraction
