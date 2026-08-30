"""Compare recommendation-breadth policies on the labelled public set.

This development-only harness uses public labels exclusively through the
official evaluator. It is not imported by the submitted Agent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.config import AgentConfig, FULL_BREADTH_POLICY, RecommendationPolicy


@dataclass(frozen=True, slots=True)
class LegacyFixedPolicy:
    """The former 1/1/3/10 schedule, retained only as a calibration control."""

    def limit_for(self, turn: int, requested: int, **_: object) -> int:
        requested = max(1, min(int(requested), 10))
        staged = (1, 1, 3)
        return min(requested, staged[turn - 1]) if 1 <= turn <= len(staged) else requested


def _summary(result: dict) -> dict:
    keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "scenario_metrics",
    )
    return {key: result[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate recommendation breadth")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/recommendation_breadth_calibration.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    policies = {
        "legacy_fixed_1_1_3_10": LegacyFixedPolicy(),
        "full_breadth": FULL_BREADTH_POLICY,
        "confidence_aware": RecommendationPolicy(),
    }
    results: dict[str, dict] = {}
    for name, policy in policies.items():
        agent = Agent(args.catalog, config=AgentConfig(recommendation_policy=policy))
        try:
            results[name] = _summary(
                evaluate(agent, samples, catalog_ids, categories, products)
            )
        finally:
            agent.close()

    baseline = results["legacy_fixed_1_1_3_10"]["recommended_technical_score"]
    adaptive = results["confidence_aware"]["recommended_technical_score"]
    payload = {
        "dataset": args.dataset,
        "policies": results,
        "confidence_aware_score_delta": round(adaptive - baseline, 6),
        "improves_public_score": adaptive > baseline,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
