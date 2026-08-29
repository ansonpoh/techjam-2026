"""Calibrate ranking-policy strength on scenario-stratified public folds.

This development-only script uses public labels through the official evaluator.
Runtime routing never receives scenario labels or ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.compare_ranking_policies import DisabledVectorIndex
from starter.agent import Agent
from starter.ranking import RankingPolicies, RankingPolicy


def policy_for_parameters(
    strength: float,
    buying_constraint_scale: float,
    buying_quality_scale: float,
    browsing_constraint_scale: float,
    browsing_quality_scale: float,
    browsing_soft_strength: float,
) -> RankingPolicies:
    return RankingPolicies(
        buying=RankingPolicy(
            constraint_scale=buying_constraint_scale,
            quality_scale=buying_quality_scale,
            vector_scale=0.0,
            hard_coverage_bonus=0.30 * strength,
            hard_exact_bonus=0.30 * strength,
            hard_missing_penalty=0.25 * strength,
            contradiction_penalty=0.50 * strength,
            budget_violation_penalty=0.50 * strength,
        ),
        browsing=RankingPolicy(
            constraint_scale=browsing_constraint_scale,
            quality_scale=browsing_quality_scale,
            soft_coverage_bonus=0.25 * browsing_soft_strength,
            soft_exact_bonus=0.35 * browsing_soft_strength,
        ),
    )


def stratified_fold(samples: list[dict], fold: str) -> list[dict]:
    if fold == "all":
        return samples
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["scenario_type"])].append(sample)
    parity = 0 if fold == "development" else 1
    return [
        sample
        for name in sorted(groups)
        for index, sample in enumerate(groups[name])
        if index % 2 == parity
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Buying-policy strength without vector API calls"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--fold",
        choices=("development", "validation", "all"),
        default="development",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    )
    parser.add_argument(
        "--buying-constraint-scales", nargs="+", type=float, default=(1.0,)
    )
    parser.add_argument(
        "--buying-quality-scales", nargs="+", type=float, default=(1.0,)
    )
    parser.add_argument(
        "--browsing-constraint-scales", nargs="+", type=float, default=(1.0,)
    )
    parser.add_argument(
        "--browsing-quality-scales", nargs="+", type=float, default=(1.0,)
    )
    parser.add_argument(
        "--browsing-soft-strengths", nargs="+", type=float, default=(0.0,)
    )
    args = parser.parse_args()

    samples = stratified_fold(load_jsonl(args.dataset), args.fold)
    catalog_ids, categories, products = catalog_index(args.catalog)
    results: list[dict] = []
    variants = [
        (
            strength,
            buying_constraint,
            buying_quality,
            browsing_constraint,
            browsing_quality,
            browsing_soft_strength,
        )
        for strength in args.strengths
        for buying_constraint in args.buying_constraint_scales
        for buying_quality in args.buying_quality_scales
        for browsing_constraint in args.browsing_constraint_scales
        for browsing_quality in args.browsing_quality_scales
        for browsing_soft_strength in args.browsing_soft_strengths
    ]
    for (
        strength,
        buying_constraint,
        buying_quality,
        browsing_constraint,
        browsing_quality,
        browsing_soft_strength,
    ) in variants:
        description = (
            f"strength={strength:g}, buying_constraint={buying_constraint:g}, "
            f"buying_quality={buying_quality:g}, "
            f"browsing_constraint={browsing_constraint:g}, "
            f"browsing_quality={browsing_quality:g}, "
            f"browsing_soft={browsing_soft_strength:g}"
        )
        print(
            f"Evaluating {description} on {args.fold} ({len(samples)} sessions)...",
            flush=True,
        )
        agent = Agent(
            args.catalog,
            ranking_policies=policy_for_parameters(
                strength,
                buying_constraint,
                buying_quality,
                browsing_constraint,
                browsing_quality,
                browsing_soft_strength,
            ),
            vector_index=DisabledVectorIndex(),
        )
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
        finally:
            agent.close()
        results.append(
            {
                "strength": strength,
                "buying_constraint_scale": buying_constraint,
                "buying_quality_scale": buying_quality,
                "browsing_constraint_scale": browsing_constraint,
                "browsing_quality_scale": browsing_quality,
                "browsing_soft_strength": browsing_soft_strength,
                "technical_score": result["recommended_technical_score"],
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "scenario_metrics": result["scenario_metrics"],
            }
        )

    results.sort(
        key=lambda item: (
            -item["technical_score"],
            -item["mrr"],
            item["mttc"],
            item["strength"],
            item["buying_constraint_scale"],
            item["buying_quality_scale"],
            item["browsing_constraint_scale"],
            item["browsing_quality_scale"],
            item["browsing_soft_strength"],
        )
    )
    print(json.dumps({"fold": args.fold, "results": results}, indent=2))


if __name__ == "__main__":
    main()
