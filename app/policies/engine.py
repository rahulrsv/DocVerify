"""Policy engine: runs every enabled rule and aggregates a demonstration outcome."""

from app.models.case import Decision, RuleResult
from app.models.common import Outcome, RuleOutcome
from app.policies.config import PolicyConfig
from app.policies.rules import RULES, PolicyContext

_SUMMARY = {
    Outcome.APPROVE: "All configured demonstration rules passed.",
    Outcome.REFER_FOR_REVIEW: "One or more rules require manual review.",
    Outcome.INSUFFICIENT_EVIDENCE: "Required evidence is missing, incomplete or out of date.",
}


def evaluate_rules(ctx: PolicyContext, config: PolicyConfig) -> list[RuleResult]:
    results: list[RuleResult] = []
    for spec in RULES:
        if spec.rule_id in config.disabled_rules:
            continue
        outcome, explanation, evidence = spec.fn(ctx, config)
        override = config.rule_overrides.get(spec.rule_id)
        results.append(
            RuleResult(
                rule_id=spec.rule_id,
                rule_version=spec.version,
                title=spec.title,
                outcome=outcome,
                fail_effect=(override.fail_effect if override and override.fail_effect else spec.fail_effect),
                severity=spec.severity,
                explanation=explanation,
                evidence=evidence,
            )
        )
    return results


def decide(results: list[RuleResult], config: PolicyConfig) -> Decision:
    failed = [r for r in results if r.outcome is RuleOutcome.FAIL]
    effects = {r.fail_effect for r in failed}
    outcome = next((o for o in config.outcome_precedence if o in effects), Outcome.APPROVE)
    # Order reasons so the ones that determined the outcome come first.
    failed.sort(key=lambda r: (config.outcome_precedence.index(r.fail_effect), r.rule_id))
    return Decision(
        outcome=outcome,
        policy_version=config.policy_version,
        summary=_SUMMARY[outcome],
        reasons=[f"[{r.rule_id}] {r.explanation}" for r in failed],
    )
