"""Inspect one labelled public session turn by turn.

This is a development-only diagnostic. It reads public labels but is never
imported by the submitted Agent.
"""

from __future__ import annotations

import argparse

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = {item["sample_id"]: item for item in load_jsonl(args.dataset)}
    sample = samples[args.sample_id]
    _, categories, products = catalog_index(args.catalog)
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    agent = Agent(args.catalog)
    session_id = "diagnostic"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories[target]), disclosed)

    print(f"target={target} scenario={sample['scenario_type']}")
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, 10)
        extended = agent.search.search(agent._sessions[session_id], limit=700)
        target_rank = next(
            (rank for rank, (asin, _) in enumerate(extended, start=1) if asin == target), None
        )
        evidence = [item.text for item in agent._sessions[session_id].evidence]
        print(
            f"turn={turn} target_rank={target_rank} ask={response['ask_attribute']}\n"
            f"  user={user_message}\n  evidence={evidence}"
        )
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )


if __name__ == "__main__":
    main()
