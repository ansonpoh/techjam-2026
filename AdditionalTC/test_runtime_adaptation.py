from __future__ import annotations

import unittest

from starter.agent import Agent
from starter.dialogue import SessionState
from starter.memory import LongTermUserProfile
from starter.ranking import DEFAULT_RANKING_POLICIES
from starter.retrieval import CatalogSearch


class RuntimeAdaptationTest(unittest.TestCase):
    def test_same_profile_id_reuses_learned_preferences(self) -> None:
        agent = Agent()
        agent.reset("one", {"profile_id": "u", "preference_tags": []})
        state = agent._sessions["one"]
        state.record_question("material")
        state.observe("For that, what matters is: leather.", 1)
        state.observe("I usually prefer leather.", 2)
        agent.reset("two", {"profile_id": "u", "preference_tags": []})

        self.assertTrue(agent._sessions["two"].long_term_profile.learned)

    def test_no_additional_preference_does_not_remove_current_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.record_question("material")
        state.observe("For that, what matters is: leather.", 1)
        state.observe("I don't have an additional preference for material.", 2)

        self.assertIn("leather", [item.text for item in state.evidence])

    def test_rejection_removes_learned_preference(self) -> None:
        profile = LongTermUserProfile("u")
        profile.observe("material", "leather", 1, durable=True, replacement=False)
        state = SessionState(user_profile={}, long_term_profile=profile)
        state.record_question("material")
        state.observe("I don't want leather.", 2)

        self.assertFalse(profile.learned)
        self.assertEqual(state.evidence[-1].source, "exclusion")

    def test_profile_bonus_is_capped_and_current_material_suppresses_it(self) -> None:
        profile = LongTermUserProfile("u")
        profile.observe("material", "leather", 1, durable=True, replacement=False)
        state = SessionState(user_profile={}, long_term_profile=profile)
        product = type("Product", (), {"token_weights": {"leather": 4.0}})()
        self.assertGreater(CatalogSearch._profile_bonus(product, state, DEFAULT_RANKING_POLICIES.browsing), 0.0)
        state.record_question("material")
        state.observe("For that, what matters is: cotton.", 2)
        self.assertEqual(CatalogSearch._profile_bonus(product, state, DEFAULT_RANKING_POLICIES.browsing), 0.0)


if __name__ == "__main__":
    unittest.main()
