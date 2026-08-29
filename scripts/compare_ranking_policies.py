"""Compare legacy and mode-aware policies without network or vector effects.

This development-only harness reads public labels through the official local
evaluator. It is not imported by the submitted Agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.ranking import LEGACY_RANKING_POLICIES
from starter.vector_index import VectorSearchResult


class DisabledVectorIndex:
    """Explicit dependency injection for deterministic, zero-API comparisons."""

    def search(
        self, structured_query: str | None, limit: int = 250
    ) -> VectorSearchResult:
        return VectorSearchResult(rows=[])

    def close(self) -> None:
        pass


def _summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "sessions"}


def _session_changes(legacy: dict, mode_aware: dict) -> list[dict]:
    previous = {item["sample_id"]: item for item in legacy["sessions"]}
    changes: list[dict] = []
    for current in mode_aware["sessions"]:
        before = previous[current["sample_id"]]
        if (
            before["best_rank"] == current["best_rank"]
            and before["first_hit_turn"] == current["first_hit_turn"]
        ):
            continue
        changes.append(
            {
                "sample_id": current["sample_id"],
                "scenario_type": current["scenario_type"],
                "legacy_rank": before["best_rank"],
                "mode_aware_rank": current["best_rank"],
                "legacy_turn": before["first_hit_turn"],
                "mode_aware_turn": current["first_hit_turn"],
            }
        )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline A/B comparison for runtime ranking policies"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output")
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Evaluate only the mode-aware policy during calibration iterations",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    print("Evaluating mode-aware policy without vectors...", flush=True)
    mode_agent = Agent(args.catalog, vector_index=DisabledVectorIndex())
    try:
        mode_aware = evaluate(
            mode_agent, samples, catalog_ids, categories, products
        )
    finally:
        mode_agent.close()

    if args.candidate_only:
        payload = {"mode_aware": _summary(mode_aware)}
        rendered = json.dumps(payload, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return

    print("Evaluating legacy policy without vectors...", flush=True)
    legacy_agent = Agent(
        args.catalog,
        ranking_policies=LEGACY_RANKING_POLICIES,
        vector_index=DisabledVectorIndex(),
    )
    try:
        legacy = evaluate(
            legacy_agent, samples, catalog_ids, categories, products
        )
    finally:
        legacy_agent.close()

    payload = {
        "legacy": _summary(legacy),
        "mode_aware": _summary(mode_aware),
        "technical_score_delta": round(
            mode_aware["recommended_technical_score"]
            - legacy["recommended_technical_score"],
            6,
        ),
        "session_changes": _session_changes(legacy, mode_aware),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
