"""Policy configuration: thresholds and rule switches live in JSON, not in prompts or code."""

import json
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from app.calculations.income import CalculationSettings
from app.models.common import DocumentType, Outcome


class Thresholds(BaseModel):
    min_recurring_annual_income: Decimal
    salary_tolerance_pct: Decimal
    max_pay_stub_age_days: int = Field(ge=1)
    variable_income_threshold_pct: Decimal


class RuleOverride(BaseModel):
    fail_effect: Outcome | None = None


class PolicyConfig(BaseModel):
    policy_version: str
    description: str = ""
    outcome_precedence: list[Outcome]
    thresholds: Thresholds
    required_document_types: list[DocumentType]
    allowed_currencies: list[str]
    disabled_rules: list[str] = Field(default_factory=list)
    rule_overrides: dict[str, RuleOverride] = Field(default_factory=dict)

    @field_validator("outcome_precedence")
    @classmethod
    def _complete_precedence(cls, value: list[Outcome]) -> list[Outcome]:
        if set(value) != set(Outcome) or len(value) != len(Outcome):
            raise ValueError("outcome_precedence must list each outcome exactly once")
        return value

    def calculation_settings(self) -> CalculationSettings:
        return CalculationSettings(variable_income_threshold_pct=self.thresholds.variable_income_threshold_pct)


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.model_validate(json.loads(path.read_text()))
