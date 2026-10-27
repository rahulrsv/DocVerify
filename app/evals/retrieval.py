"""Retrieval evaluation: does policy-text lookup return the right handbook passage?

Two query sets, both offline:
- rule_citations: every failed-rule explanation produced by running the golden cases'
  labeled extractions through the real rules, queried the way the pipeline queries
  (rule title + explanation). The expected passage is the failed rule's own section.
- explanation_only: the same explanations without the rule title (a harder, ID-free check).
- reviewer_questions: hand-written questions in data/retrieval/reviewer_questions.jsonl,
  phrased the way a reviewer would ask, each labeled with the passage that answers it.

Usage: python -m app.evals.retrieval [--backend bm25|embedding] [--fail-under-thresholds]
"""

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.evals.dataset import load_dataset
from app.evals.service import gold_rule_results
from app.models.common import RuleOutcome
from app.policies.config import load_policy
from app.retrieval.factory import build_retriever
from app.retrieval.retriever import PolicyRetriever

RETRIEVAL_DIR = PROJECT_ROOT / "data" / "retrieval"
K = 3


def _score(retriever: PolicyRetriever, queries: list[dict[str, str]]) -> dict[str, Any]:
    hit1 = hit3 = 0
    rr = 0.0
    misses = []
    for q in queries:
        ranked = [h.passage.passage_id for h in retriever.search(q["query"], k=K)]
        rank = ranked.index(q["expected"]) + 1 if q["expected"] in ranked else None
        hit1 += rank == 1
        hit3 += rank is not None
        rr += 1 / rank if rank else 0.0
        if rank != 1:
            misses.append({**q, "retrieved": ranked})
    n = len(queries)
    return {"queries": n, "hit_at_1": round(hit1 / n, 4), "hit_at_3": round(hit3 / n, 4), "mrr": round(rr / n, 4), "misses": misses}


def rule_queries(settings: Settings, with_title: bool) -> list[dict[str, str]]:
    policy = load_policy(settings.policy_path)
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for case in load_dataset(settings.golden_dir).cases:
        for r in gold_rule_results(case, policy):
            if r.outcome is RuleOutcome.FAIL:
                query = f"{r.title}. {r.explanation}" if with_title else r.explanation
                seen.setdefault((r.rule_id, query), {"id": case.case_id, "query": query, "expected": r.rule_id})
    return list(seen.values())


def reviewer_queries(path: Path = RETRIEVAL_DIR / "reviewer_questions.jsonl") -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(settings: Settings) -> dict[str, Any]:
    retriever = build_retriever(settings, load_policy(settings.policy_path))
    if retriever is None:
        raise ValueError("retrieval backend is 'none'")
    metrics = {
        "rule_citations": _score(retriever, rule_queries(settings, with_title=True)),
        "explanation_only": _score(retriever, rule_queries(settings, with_title=False)),
        "reviewer_questions": _score(retriever, reviewer_queries()),
    }
    limits = json.loads((RETRIEVAL_DIR / "thresholds.json").read_text()).get(settings.retrieval_backend, {})
    checks = []
    for name, bound in limits.items():
        group, metric = name.split(".")
        value = metrics[group][metric]
        checks.append({"metric": name, "value": value, "min": bound["min"], "passed": value >= bound["min"]})
    return {"created_at": datetime.now(UTC).isoformat(), "retrieval_version": retriever.version, "passages": len(retriever.handbook.passages), "metrics": metrics, "thresholds": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["bm25", "embedding"], default=None)
    parser.add_argument("--fail-under-thresholds", action="store_true", help="Exit 1 when a threshold for this backend is missed")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.backend:
        settings = dataclasses.replace(settings, retrieval_backend=args.backend)
    report = run(settings)

    print(f"RETRIEVAL EVAL | {report['retrieval_version']} | {report['passages']} handbook passages")
    for group, m in report["metrics"].items():
        print(f"  {group:<20} n={m['queries']:<3} hit@1 {m['hit_at_1']:.1%}  hit@3 {m['hit_at_3']:.1%}  MRR {m['mrr']:.3f}")
    for group, m in report["metrics"].items():
        for miss in m["misses"]:
            print(f"  miss [{group}] {miss['id']}: {miss['query'][:70]!r} expected {miss['expected']} got {miss['retrieved']}")
    for t in report["thresholds"]:
        print(f"  [{'PASS' if t['passed'] else 'FAIL'}] {t['metric']} = {t['value']} (>= {t['min']})")
    if not report["thresholds"]:
        print("  no thresholds configured for this backend")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / f"retrieval_latest_{settings.retrieval_backend}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")
    return 1 if args.fail_under_thresholds and not all(t["passed"] for t in report["thresholds"]) else 0


if __name__ == "__main__":
    sys.exit(main())
