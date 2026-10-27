"""Policy handbook corpus: a Markdown file split into one passage per `##` section."""

import re
from dataclasses import dataclass
from pathlib import Path

from app.policies.config import PolicyConfig

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_SECTION = re.compile(r"^## \[(?P<id>[A-Z0-9_]+)\] (?P<title>.+)$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str

    @property
    def rule_id(self) -> str | None:
        """Sections named after a rule (R###_...) explain that rule; GEN_* sections are general."""
        return self.passage_id if re.fullmatch(r"R\d{3}_[A-Z0-9_]+", self.passage_id) else None

    @property
    def searchable(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Handbook:
    version: str
    policy_version: str
    passages: list[Passage]

    def get(self, passage_id: str) -> Passage | None:
        return next((p for p in self.passages if p.passage_id == passage_id), None)


def _policy_values(policy: PolicyConfig) -> dict[str, str]:
    t = policy.thresholds
    return {
        "min_recurring_annual_income": f"{t.min_recurring_annual_income} {'/'.join(policy.allowed_currencies)}",
        "salary_tolerance_pct": str(t.salary_tolerance_pct),
        "max_pay_stub_age_days": str(t.max_pay_stub_age_days),
        "variable_income_threshold_pct": str(t.variable_income_threshold_pct),
        "required_document_types": ", ".join(d.value for d in policy.required_document_types),
        "allowed_currencies": ", ".join(policy.allowed_currencies),
        "outcome_precedence": " > ".join(o.value for o in policy.outcome_precedence),
    }


def parse_handbook(markdown: str, policy: PolicyConfig) -> Handbook:
    front = _FRONT_MATTER.match(markdown)
    if not front:
        raise ValueError("handbook is missing its front matter block")
    meta = dict(line.split(":", 1) for line in front.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    if meta.get("policy_version") != policy.policy_version:
        raise ValueError(f"handbook targets policy {meta.get('policy_version')!r} but the active policy is {policy.policy_version!r}")

    values = _policy_values(policy)

    def fill(text: str) -> str:
        def sub(match: re.Match[str]) -> str:
            if match.group(1) not in values:
                raise ValueError(f"unknown handbook placeholder {{{match.group(1)}}}")
            return values[match.group(1)]

        return _PLACEHOLDER.sub(sub, text)

    body = markdown[front.end():]
    headings = list(_SECTION.finditer(body))
    passages = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = " ".join(body[heading.end():end].split())
        passages.append(Passage(heading.group("id"), heading.group("title").strip(), fill(text)))
    ids = [p.passage_id for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("handbook passage IDs must be unique")
    return Handbook(version=meta["handbook_version"], policy_version=meta["policy_version"], passages=passages)


def load_handbook(path: Path, policy: PolicyConfig) -> Handbook:
    return parse_handbook(path.read_text(), policy)
