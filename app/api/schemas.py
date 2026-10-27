"""API request/response models that are not domain models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.case import CaseResult


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str
    prompt_version: str
    policy_version: str
    schema_version: str


class AuditResponse(BaseModel):
    case_id: str
    runs: list[dict[str, Any]]
    events: list[dict[str, Any]]
    chain_verified: bool


class EvalRunRequest(BaseModel):
    provider: Literal["mock", "openai"] = "mock"
    update_baseline: bool = False
    case_ids: list[str] | None = Field(default=None, description="Optional subset of golden case IDs")


class ErrorResponse(BaseModel):
    detail: str


ProcessResponse = CaseResult
