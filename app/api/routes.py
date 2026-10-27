"""HTTP handlers. Kept thin: validate the request, call the pipeline or store, map errors."""

import re
import threading
import uuid
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import AuditResponse, ErrorResponse, EvalRunRequest, HealthResponse
from app.audit.store import AuditStore
from app.core.config import Settings
from app.core.pipeline import CaseProcessor, DocumentInput
from app.documents.loader import DocumentRejected, validate_upload
from app.evals.service import EvaluationError, load_latest_report, run_evaluation
from app.models.case import CaseResult
from app.models.common import SCHEMA_VERSION

router = APIRouter()
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_eval_lock = threading.Lock()
_REJECTION_STATUS = {
    "unsupported_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "empty": status.HTTP_400_BAD_REQUEST,
}


def _processor(request: Request) -> CaseProcessor:
    return request.app.state.processor


def _audit(request: Request) -> AuditStore:
    return request.app.state.audit


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _check_case_id(case_id: str) -> None:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "case_id must be 1-64 characters of letters, digits, '-' or '_'")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    proc = _processor(request)
    return HealthResponse(
        status="ok",
        provider=proc.provider.name,
        model=proc.provider.model,
        prompt_version=proc.provider.prompt_version,
        policy_version=proc.policy.policy_version,
        schema_version=SCHEMA_VERSION,
    )


@router.post(
    "/documents/process",
    response_model=CaseResult,
    status_code=status.HTTP_201_CREATED,
    tags=["processing"],
    responses={200: {"description": "Idempotent replay of an identical earlier run"}, 400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Process one or more related synthetic documents as a single case",
)
async def process_documents(
    request: Request,
    response: Response,
    files: Annotated[list[UploadFile], File(description="One or more .pdf or .txt synthetic documents")],
    case_id: Annotated[str | None, Form(description="Synthetic case ID; generated if omitted")] = None,
    as_of_date: Annotated[date | None, Form(description="Evaluation date for recency checks (default: today, UTC)")] = None,
    force: Annotated[bool, Form(description="Reprocess even if an identical run exists")] = False,
) -> CaseResult:
    settings = _settings(request)
    case_id = case_id or f"case_{uuid.uuid4().hex[:10]}"
    _check_case_id(case_id)
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one file is required")
    if len(files) > settings.max_files_per_case:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"At most {settings.max_files_per_case} files per case")

    documents: list[DocumentInput] = []
    for upload in files:
        # Starlette has already spooled the upload; reading limit+1 bytes keeps oversized files out of memory here.
        content = await upload.read(settings.max_upload_bytes + 1)
        try:
            validate_upload(upload.filename or "", content, settings.max_upload_bytes)
        except DocumentRejected as exc:
            raise HTTPException(_REJECTION_STATUS[exc.reason], str(exc)) from exc
        documents.append(DocumentInput(filename=upload.filename or "upload", content=content))

    # Processing is synchronous (PDF parsing, provider calls with retries); keep it off the event loop.
    result = await run_in_threadpool(_processor(request).process_case, case_id, documents, as_of_date=as_of_date, force=force)
    if result.idempotent_replay:
        response.status_code = status.HTTP_200_OK
    return result


@router.get("/cases/{case_id}", response_model=CaseResult, tags=["cases"], responses={404: {"model": ErrorResponse}})
def get_case(case_id: str, request: Request) -> CaseResult:
    _check_case_id(case_id)
    result = _audit(request).latest_run(case_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Case '{case_id}' not found")
    return result


@router.get("/cases", tags=["cases"])
def list_cases(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    return _audit(request).list_cases(min(max(limit, 1), 500))


@router.get("/cases/{case_id}/audit", response_model=AuditResponse, tags=["cases"], responses={404: {"model": ErrorResponse}})
def get_case_audit(case_id: str, request: Request) -> AuditResponse:
    _check_case_id(case_id)
    audit = _audit(request)
    events = audit.events_for_case(case_id)
    if not events:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No audit history for case '{case_id}'")
    ok, _ = audit.verify_chain()
    return AuditResponse(case_id=case_id, runs=audit.runs_for_case(case_id), events=events, chain_verified=ok)


@router.get("/evals/latest", tags=["evaluation"], responses={404: {"model": ErrorResponse}})
def get_latest_eval(request: Request) -> dict[str, Any]:
    report = load_latest_report(_settings(request).reports_dir)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No evaluation has been run yet")
    return report


@router.post("/evals/run", tags=["evaluation"], responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def trigger_eval(request: Request, body: EvalRunRequest | None = None) -> dict[str, Any]:
    body = body or EvalRunRequest()
    if not _eval_lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "An evaluation is already running")
    try:
        settings = _settings(request)
        report = run_evaluation(
            settings,
            provider_name=body.provider,
            update_baseline=body.update_baseline,
            case_ids=body.case_ids,
            audit=_audit(request),
        )
    except EvaluationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    finally:
        _eval_lock.release()
    return report["summary"]
