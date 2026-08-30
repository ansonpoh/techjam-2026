"""Calibrate recommendation breadth on disjoint development/validation folds.

The expensive ranking trajectory is collected once with full breadth. Breadth
does not affect retrieval, question selection, or simulated customer replies,
so candidate policies can then be replayed exactly without rebuilding the
catalog index for every grid point. Ground-truth identifiers are used only in
this development-only script and are never imported by the runtime agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.config import AgentConfig, RecommendationPolicy


@dataclass(frozen=True, slots=True)
class FixedPolicy:
    """A fixed staged width, with top-10 after its final staged turn."""

    widths: tuple[int, ...]

    def limit_for(self, turn: int, requested: int, **_: object) -> int:
        width = self.widths[turn - 1] if 1 <= turn <= len(self.widths) else requested
        return min(max(1, int(requested)), width)


class RecordingPolicy:
    """Expose full rankings while retaining inputs to the runtime policy."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def limit_for(self, turn: int, requested: int, **features: object) -> int:
        self.calls.append({"turn": turn, "requested": requested, **features})
        return min(max(1, int(requested)), TOP_K)


def _collect_trajectories(
    catalog_path: str,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> list[dict]:
    recorder = RecordingPolicy()
    agent = Agent(
        catalog_path,
        config=AgentConfig(recommendation_policy=recorder),  # type: ignore[arg-type]
    )
    trajectories: list[dict] = []
    try:
        for sample in samples:
            session_id = f"breadth_calibration_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(
                effective_sample, coarse_category(categories.get(target, [])), disclosed
            )
            turns: list[dict] = []
            for turn in range(1, MAX_TURNS + 1):
                before = len(recorder.calls)
                response = agent.respond(session_id, user_message, turn, TOP_K)
                if len(recorder.calls) != before + 1:
                    raise RuntimeError("recommendation policy was not called exactly once")
                inputs = recorder.calls[-1]
                ranked = normalize_recommendations(
                    response.get("recommendations"), catalog_ids
                )
                turns.append({
                    "eligible": override_applied,
                    "ranked": ranked,
                    "policy_inputs": inputs,
                })
                if turn == MAX_TURNS:
                    break
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get(
                        "message", "Actually, please ignore my earlier preference."
                    ))
                else:
                    user_message, boundary_used = customer_reply(
                        effective_sample, response.get("ask_attribute"), disclosed,
                        boundary_used,
                    )
            trajectories.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "target": target,
                "turns": turns,
            })
    finally:
        agent.close()
    return trajectories


def _evaluate_policy(policy: object, trajectories: list[dict]) -> dict:
    sessions: list[dict] = []
    for trajectory in trajectories:
        target = trajectory["target"]
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn_number, turn in enumerate(trajectory["turns"], start=1):
            inputs = turn["policy_inputs"]
            limit = policy.limit_for(  # type: ignore[attr-defined]
                inputs["turn"], inputs["requested"],
                **{key: value for key, value in inputs.items()
                   if key not in {"turn", "requested"}},
            )
            ranked = turn["ranked"][:limit]
            if turn["eligible"] and target in ranked:
                hit_turn = turn_number
                best_rank = ranked.index(target) + 1
                break
        sessions.append({
            "scenario_type": trajectory["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    return _metrics(sessions)


def _metrics(sessions: list[dict]) -> dict:
    count = len(sessions)
    hit_rate = sum(item["hit"] for item in sessions) / count
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
        for item in sessions
    )
    efficiency = (11.0 - mttc) / 10.0
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _split(trajectories: list[dict], validation_fraction: float) -> tuple[list, list]:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        by_scenario[trajectory["scenario_type"]].append(trajectory)
    development: list[dict] = []
    validation: list[dict] = []
    for _, group in sorted(by_scenario.items()):
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"breadth-v1\0{item['sample_id']}".encode()
            ).digest(),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation.extend(ordered[:validation_count])
        development.extend(ordered[validation_count:])
    return development, validation


def _candidate_policies() -> dict[str, object]:
    policies: dict[str, object] = {
        "fixed_1_1_3": FixedPolicy((1, 1, 3)),
        "fixed_1_2_3": FixedPolicy((1, 2, 3)),
        "fixed_1_3_5": FixedPolicy((1, 3, 5)),
        "fixed_1_1_5": FixedPolicy((1, 1, 5)),
        "fixed_1_3_10": FixedPolicy((1, 3, 10)),
        "full_breadth": FixedPolicy(()),
    }
    for high_margin in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18):
        for low_margin in (0.0, 0.005, 0.01):
            for low_entropy in (0.65, 0.72, 0.80, 0.90):
                for high_entropy in (0.95, 0.98, 1.0):
                    for horizon in (1, 2, 3):
                        for width in (2, 3, 5):
                            name = (
                                f"adaptive_hm{high_margin:g}_lm{low_margin:g}"
                                f"_le{low_entropy:g}_he{high_entropy:g}"
                                f"_h{horizon}_w{width}"
                            )
                            policies[name] = RecommendationPolicy(
                                high_margin=high_margin, low_margin=low_margin,
                                low_entropy=low_entropy, high_entropy=high_entropy,
                                clarification_horizon=horizon, moderate_width=width,
                            )
    return policies


def _policy_parameters(policy: object) -> dict:
    if isinstance(policy, FixedPolicy):
        return {"kind": "fixed", "widths": list(policy.widths)}
    return {
        "kind": "adaptive",
        "high_margin": policy.high_margin,
        "low_margin": policy.low_margin,
        "low_entropy": policy.low_entropy,
        "high_entropy": policy.high_entropy,
        "clarification_horizon": policy.clarification_horizon,
        "moderate_width": policy.moderate_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate recommendation breadth")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/recommendation_breadth_calibration.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction must be between 0.05 and 0.5")

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    trajectories = _collect_trajectories(
        args.catalog, samples, catalog_ids, categories, products
    )
    development, validation = _split(trajectories, args.validation_fraction)
    policies = _candidate_policies()
    development_results = {
        name: _evaluate_policy(policy, development)
        for name, policy in policies.items()
    }
    selected_name = max(
        policies,
        key=lambda name: (
            development_results[name]["recommended_technical_score"],
            development_results[name]["mrr"], -len(name),
        ),
    )
    baseline_name = "fixed_1_1_3"
    selected = policies[selected_name]
    validation_baseline = _evaluate_policy(policies[baseline_name], validation)
    validation_selected = _evaluate_policy(selected, validation)
    full_baseline = _evaluate_policy(policies[baseline_name], trajectories)
    full_selected = _evaluate_policy(selected, trajectories)
    fixed_results = {
        name: {
            "development": development_results[name],
            "validation": _evaluate_policy(policy, validation),
        }
        for name, policy in policies.items() if isinstance(policy, FixedPolicy)
    }
    payload = {
        "dataset": args.dataset,
        "split": {
            "method": "scenario-stratified deterministic SHA-256 holdout",
            "development_samples": len(development),
            "validation_samples": len(validation),
            "validation_fraction": args.validation_fraction,
        },
        "candidate_count": len(policies),
        "fixed_policy_results": fixed_results,
        "selected_policy": {
            "name": selected_name,
            "parameters": _policy_parameters(selected),
            "development": development_results[selected_name],
            "validation": validation_selected,
            "full_public_set": full_selected,
        },
        "legacy_policy": {
            "validation": validation_baseline,
            "full_public_set": full_baseline,
        },
        "validation_score_delta": round(
            validation_selected["recommended_technical_score"]
            - validation_baseline["recommended_technical_score"], 6
        ),
        "validation_improves": (
            validation_selected["recommended_technical_score"]
            > validation_baseline["recommended_technical_score"]
        ),
        "full_public_score_delta": round(
            full_selected["recommended_technical_score"]
            - full_baseline["recommended_technical_score"], 6
        ),
        "full_public_improves": (
            full_selected["recommended_technical_score"]
            > full_baseline["recommended_technical_score"]
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
