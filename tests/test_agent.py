from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.dialogue import SessionState
from starter.question_planner import AdaptiveQuestionPlanner
from starter.retrieval import CatalogSearch, Evidence


class DialogueStateTest(unittest.TestCase):
    def test_free_form_answer_does_not_require_simulator_wording(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        state.observe("Breathable cotton would be ideal for warm weather", 2)
        self.assertIn(
            "breathable cotton would be ideal for warm weather",
            [item.text.lower() for item in state.evidence],
        )

    def test_accumulates_constraints_and_removes_opening_preference_on_override(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes. I prefer red.", 1)
        state.observe("For that, what matters is: leather; wide width.", 2)
        state.observe("Actually, ignore my earlier preference. What I need is: black.", 3)
        evidence = [item.text.lower() for item in state.evidence]
        self.assertIn("shoes", evidence)
        self.assertIn("leather", evidence)
        self.assertIn("wide width", evidence)
        self.assertIn("black", evidence)
        self.assertNotIn("i prefer red", evidence)

    def test_no_preference_is_not_positive_search_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Jackets, but I'm still exploring.", 1)
        state.record_question("other")
        state.observe("I don't have a preference for other; please use your judgment.", 2)
        self.assertEqual([item.text.lower() for item in state.evidence], ["jackets"])


class AdaptiveQuestionPlannerTest(unittest.TestCase):
    @staticmethod
    def _candidate(title: str, score: float) -> dict:
        return {
            "parent_asin": title,
            "title": title,
            "categories": "Clothing Shirts",
            "features": "",
            "details": "",
            "store": "Example",
            "description": "",
            "price": "50",
            "_rank_score": score,
        }

    def test_question_attribute_changes_with_candidate_differences(self) -> None:
        planner = AdaptiveQuestionPlanner()
        material_state = SessionState(user_profile={})
        material_candidates = [
            self._candidate("cotton shirt", 20.0),
            self._candidate("leather shirt", 14.0),
            self._candidate("polyester shirt", 10.0),
        ]
        material_attribute, material_question = planner.choose(
            material_state, material_candidates, 1
        )

        color_state = SessionState(user_profile={})
        color_candidates = [
            self._candidate("red shirt", 20.0),
            self._candidate("blue shirt", 14.0),
            self._candidate("green shirt", 10.0),
        ]
        color_attribute, color_question = planner.choose(color_state, color_candidates, 1)

        self.assertEqual(material_attribute, "material")
        self.assertEqual(color_attribute, "color")
        self.assertIn("cotton", material_question)
        self.assertIn("red", color_question)
        self.assertNotEqual(material_question, color_question)


class AgentRetrievalTest(unittest.TestCase):
    def test_budget_proximity_is_scored_without_a_network_model(self) -> None:
        evidence = [Evidence("budget around $50", 3.0, "clarification", 2)]
        close = CatalogSearch._price_score({"price": "52"}, evidence)
        far = CatalogSearch._price_score({"price": "120"}, evidence)
        self.assertGreater(close, far)

    def test_conversation_reranks_exact_constraint_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": "A", "title": "Everyday Boot", "categories": ["Shoes"],
                    "features": ["synthetic", "standard width"], "details": {},
                    "store": "Example", "description": [], "price": 40,
                    "average_rating": 4.8, "rating_number": 500,
                },
                {
                    "parent_asin": "B", "title": "Trail Boot", "categories": ["Shoes"],
                    "features": ["full grain leather", "wide width"], "details": {},
                    "store": "Example", "description": [], "price": 60,
                    "average_rating": 4.0, "rating_number": 10,
                },
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            agent = Agent(catalog)
            agent.reset("s", {})
            agent.respond("s", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
            response = agent.respond(
                "s", "For that, what matters is: full grain leather; wide width.", 2, 10
            )
            self.assertEqual(response["recommendations"][0]["parent_asin"], "B")


if __name__ == "__main__":
    unittest.main()
