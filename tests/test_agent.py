from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.config import (
    AgentConfig,
    FULL_BREADTH_POLICY,
    RecommendationPolicy,
)
from starter.dialogue import Evidence, SessionState
from starter.product_features import FIELD_WEIGHTS, ProductFeatureStore, terms
from starter.question_planner import AdaptiveQuestionPlanner
from starter.ranking import (
    DEFAULT_RANKING_POLICIES,
    LEGACY_POLICY,
    IntentRouter,
    RankingMode,
)
from starter.retrieval import CatalogSearch, QUALITY_REVIEW_WEIGHT


def _legacy_constraint_score(
    product: dict, evidence: list[Evidence], user_profile: dict | None = None
) -> float:
    field_tokens = {
        field: set(terms(str(product.get(field) or "")))
        for field in FIELD_WEIGHTS
    }
    normalized_fields = {
        field: " ".join(terms(str(product.get(field) or "")))
        for field in FIELD_WEIGHTS
    }
    score = 0.0
    for item in evidence:
        query_terms = list(dict.fromkeys(terms(item.text)))
        if not query_terms:
            continue
        matched_weight = 0.0
        matched_terms = 0
        for token in query_terms:
            best_field_weight = max(
                (
                    weight
                    for field, weight in FIELD_WEIGHTS.items()
                    if token in field_tokens[field]
                ),
                default=0.0,
            )
            matched_weight += best_field_weight
            matched_terms += int(best_field_weight > 0.0)
        coverage = matched_terms / len(query_terms)
        field_affinity = matched_weight / (
            len(query_terms) * max(FIELD_WEIGHTS.values())
        )
        score += item.weight * (1.9 * coverage + 0.4 * field_affinity)
        normalized_query = " ".join(query_terms)
        if len(query_terms) >= 2 and any(
            normalized_query in value for value in normalized_fields.values()
        ):
            score += item.weight * min(2.0, 0.55 + 0.22 * len(query_terms))
        if coverage >= 0.999:
            score += item.weight * 0.45
    tags = user_profile.get("preference_tags") if isinstance(user_profile, dict) else None
    if isinstance(tags, list) and tags:
        preference_terms = {token for tag in tags for token in terms(str(tag))}
        product_terms = set().union(*field_tokens.values())
        if preference_terms:
            score += 0.45 * len(preference_terms & product_terms) / len(preference_terms)
    return score


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


class IntentRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_open_ended_session_stays_in_browsing_mode_as_preferences_accumulate(
        self,
    ) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        state.record_question("material")
        state.observe("For that, what matters is: breathable cotton.", 2)

        decision = self.router.route(state)

        self.assertEqual(decision.mode, RankingMode.BROWSING)
        self.assertEqual(decision.reasons, ("open_ended_start",))

    def test_explicit_requirement_routes_to_buying_without_scenario_labels(self) -> None:
        state = SessionState(user_profile={})
        state.observe(
            "I'm looking for Shoes. A key requirement is: waterproof wide width.",
            1,
        )

        decision = self.router.route(state)

        self.assertEqual(decision.mode, RankingMode.BUYING)
        self.assertIn("explicit_requirement", decision.reasons)

    def test_open_session_can_transition_to_buying_after_explicit_override(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        self.assertEqual(self.router.route(state).mode, RankingMode.BROWSING)

        state.observe("Actually, what I need is: black leather.", 2)

        self.assertEqual(self.router.route(state).mode, RankingMode.BUYING)


class AdaptiveQuestionPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_store = ProductFeatureStore()
        self.planner = AdaptiveQuestionPlanner(self.feature_store)

    def _candidate(self, title: str, score: float) -> dict:
        candidate = {
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
        candidate["_features"] = self.feature_store.add(
            title,
            {field: str(candidate.get(field) or "") for field in FIELD_WEIGHTS},
            price=candidate["price"],
        )
        return candidate

    def test_question_attribute_changes_with_candidate_differences(self) -> None:
        material_state = SessionState(user_profile={})
        material_state.asked_attributes.extend(["other", "other"])
        material_candidates = [
            self._candidate("cotton shirt", 20.0),
            self._candidate("leather shirt", 14.0),
            self._candidate("polyester shirt", 10.0),
        ]
        material_attribute, material_question = self.planner.choose(
            material_state, material_candidates, 1
        )

        color_state = SessionState(user_profile={})
        color_state.asked_attributes.extend(["other", "other"])
        color_candidates = [
            self._candidate("red shirt", 20.0),
            self._candidate("blue shirt", 14.0),
            self._candidate("green shirt", 10.0),
        ]
        color_attribute, color_question = self.planner.choose(
            color_state, color_candidates, 1
        )

        self.assertEqual(material_attribute, "material")
        self.assertEqual(color_attribute, "color")
        self.assertIn("cotton", material_question)
        self.assertIn("red", color_question)
        self.assertNotEqual(material_question, color_question)

    def test_early_question_prioritizes_must_have_without_repeating_boundary(self) -> None:
        state = SessionState(user_profile={})
        candidates = [
            self._candidate("cotton shirt", 20.0),
            self._candidate("leather shirt", 14.0),
        ]

        attribute, _ = self.planner.choose(state, candidates, 1)
        self.assertEqual(attribute, "other")

        state.no_preference_attributes.add("other")
        next_attribute, _ = self.planner.choose(state, candidates, 2)
        self.assertNotEqual(next_attribute, "other")


class AgentRetrievalTest(unittest.TestCase):
    @staticmethod
    def _catalog_rows() -> list[dict]:
        return [
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

    @classmethod
    def _write_catalog(cls, directory: str) -> Path:
        catalog = Path(directory) / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(row) + "\n" for row in cls._catalog_rows()),
            encoding="utf-8",
        )
        return catalog

    @staticmethod
    def _features(
        store: ProductFeatureStore,
        parent_asin: str,
        *,
        title: str = "",
        features: str = "",
        price: float = 50.0,
        average_rating: float = 4.0,
        rating_number: int = 1,
    ):
        fields = {field: "" for field in FIELD_WEIGHTS}
        fields.update({"title": title, "features": features})
        return store.add(
            parent_asin,
            fields,
            price=price,
            average_rating=average_rating,
            rating_number=rating_number,
        )

    def test_precomputed_features_are_reused_and_read_only(self) -> None:
        store = ProductFeatureStore()
        product = self._features(
            store,
            "A",
            title="Cotton running shirt",
            features="breathable lightweight fabric",
        )
        self.assertIs(store.get("A"), product)
        self.assertEqual(len(store), 1)
        with self.assertRaises(TypeError):
            product.token_weights[0] = 99.0  # type: ignore[index]
        with self.assertRaises(ValueError):
            self._features(store, "A", title="duplicate")

    def test_feature_cache_reuses_entries_and_evicts_least_recently_used(self) -> None:
        store = ProductFeatureStore(max_size=2)
        fields = {field: "cotton shirt" for field in FIELD_WEIGHTS}
        first = store.get_or_add("A", fields)
        self.assertIs(store.get_or_add("A", fields), first)
        store.get_or_add("B", fields)
        store.get_or_add("C", fields)
        info = store.cache_info()
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.misses, 3)
        self.assertEqual(info.evictions, 1)
        self.assertEqual(info.current_size, 2)
        with self.assertRaises(KeyError):
            store.get("A")

    def test_cached_constraint_score_matches_previous_formula(self) -> None:
        store = ProductFeatureStore()
        raw_product = {
            "title": "Trail Boot",
            "categories": "Clothing Shoes Hiking Boots",
            "features": "full grain leather waterproof wide width",
            "details": "material leather color brown",
            "store": "Example",
            "description": "comfortable outdoor walking boot",
        }
        product = store.add(
            "B",
            raw_product,
            price=60,
            average_rating=4.4,
            rating_number=120,
        )
        evidence = [
            Evidence("Hiking Boots", 1.4, "category", 1),
            Evidence("full grain leather; wide width", 3.3, "clarification", 2),
            Evidence("unseen query token", 2.0, "clarification", 3),
        ]
        profile = {"preference_tags": ["comfort", "durability"]}
        query = store.compile_query(evidence, profile)
        cached = CatalogSearch._constraint_score(product, query)
        previous = _legacy_constraint_score(raw_product, evidence, profile)
        self.assertAlmostEqual(cached, previous, places=12)

    def test_popularity_weight_is_reduced_and_bounded(self) -> None:
        store = ProductFeatureStore()
        obscure = CatalogSearch._quality_tiebreak(
            self._features(store, "obscure", rating_number=1)
        )
        popular = CatalogSearch._quality_tiebreak(
            self._features(store, "popular", rating_number=10000)
        )
        self.assertLess(QUALITY_REVIEW_WEIGHT, 1.20)
        self.assertLess(popular - obscure, 9.5)

    def test_budget_proximity_is_scored_without_a_network_model(self) -> None:
        store = ProductFeatureStore()
        evidence = [Evidence("budget around $50", 3.0, "clarification", 2)]
        query = store.compile_query(evidence)
        close = CatalogSearch._price_score(
            self._features(store, "close", price=52), query
        )
        far = CatalogSearch._price_score(
            self._features(store, "far", price=120), query
        )
        self.assertGreater(close, far)

    def test_buying_policy_penalizes_hard_constraint_contradictions(self) -> None:
        store = ProductFeatureStore()
        matching_raw = {
            "parent_asin": "MATCH",
            "title": "Black leather trail boot",
            "features": "wide width waterproof",
            "categories": "Shoes",
            "details": "",
            "store": "Example",
            "description": "",
        }
        conflicting_raw = {
            **matching_raw,
            "parent_asin": "CONFLICT",
            "title": "Red synthetic trail boot",
        }
        matching = store.add("MATCH", matching_raw)
        conflicting = store.add("CONFLICT", conflicting_raw)
        query = store.compile_query(
            [Evidence("black leather", 3.8, "hard_constraint", 1)]
        )
        policy = DEFAULT_RANKING_POLICIES.buying

        matching_score = CatalogSearch._constraint_fit_adjustment(
            matching,
            store.question_features(matching_raw),
            query,
            policy,
        )
        conflicting_score = CatalogSearch._constraint_fit_adjustment(
            conflicting,
            store.question_features(conflicting_raw),
            query,
            policy,
        )

        self.assertGreater(matching_score, 0.0)
        self.assertLess(conflicting_score, 0.0)
        self.assertGreater(matching_score, conflicting_score)

    def test_legacy_policy_adds_no_mode_specific_adjustment(self) -> None:
        store = ProductFeatureStore()
        raw = {
            "parent_asin": "A",
            "title": "Red synthetic boot",
            "features": "standard width",
            "categories": "Shoes",
            "details": "",
            "store": "Example",
            "description": "",
        }
        product = store.add("A", raw, price=120)
        query = store.compile_query(
            [
                Evidence("black leather", 3.8, "hard_constraint", 1),
                Evidence("under $50", 3.0, "clarification", 2),
            ]
        )

        self.assertEqual(
            CatalogSearch._constraint_fit_adjustment(
                product,
                store.question_features(raw),
                query,
                LEGACY_POLICY,
            ),
            0.0,
        )
        self.assertEqual(
            CatalogSearch._budget_violation_adjustment(
                product, query, LEGACY_POLICY
            ),
            0.0,
        )

    def test_buying_policy_penalizes_explicit_budget_violation(self) -> None:
        store = ProductFeatureStore()
        query = store.compile_query(
            [Evidence("under $50", 3.0, "hard_constraint", 1)]
        )
        policy = DEFAULT_RANKING_POLICIES.buying
        affordable = self._features(store, "affordable", price=45)
        expensive = self._features(store, "expensive", price=100)

        affordable_score = CatalogSearch._budget_violation_adjustment(
            affordable, query, policy
        )
        expensive_score = CatalogSearch._budget_violation_adjustment(
            expensive, query, policy
        )

        self.assertEqual(affordable_score, 0.0)
        self.assertLess(expensive_score, affordable_score)

    def test_search_result_reports_inferred_ranking_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            agent = Agent(catalog)
            agent.reset("browse", {})
            agent.respond(
                "browse", "I'm looking for Shoes, but I'm still exploring.", 1, 10
            )
            browsing = agent.search.search_with_context(agent._sessions["browse"])

            agent.reset("buy", {})
            agent.respond(
                "buy",
                "I'm looking for Shoes. A key requirement is: wide width.",
                1,
                10,
            )
            buying = agent.search.search_with_context(agent._sessions["buy"])

            self.assertEqual(browsing.ranking_mode, RankingMode.BROWSING)
            self.assertEqual(buying.ranking_mode, RankingMode.BUYING)

    def test_default_recommendation_policy_stages_early_breadth(self) -> None:
        policy = RecommendationPolicy()

        self.assertEqual([policy.limit_for(turn, 10) for turn in range(1, 5)], [1, 1, 3, 10])
        self.assertEqual(policy.limit_for(3, 2), 2)

    def test_full_breadth_policy_is_available_for_controlled_comparisons(self) -> None:
        config = AgentConfig(recommendation_policy=FULL_BREADTH_POLICY)

        self.assertEqual(config.recommendation_policy.limit_for(1, 10), 10)

    def test_default_agent_does_not_construct_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            with patch("starter.retrieval.CatalogVectorIndex") as vector_type:
                agent = Agent(catalog)
                try:
                    self.assertIsNone(agent.search.vector_index)
                    vector_type.assert_not_called()
                finally:
                    agent.close()

    def test_vector_reranker_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            config = AgentConfig(enable_vector_reranker=True)
            with patch("starter.retrieval.CatalogVectorIndex") as vector_type:
                agent = Agent(catalog, config=config)
                try:
                    vector_type.assert_called_once_with(catalog)
                    self.assertIs(agent.search.vector_index, vector_type.return_value)
                finally:
                    agent.close()

    def test_conversation_reranks_exact_constraint_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            agent = Agent(catalog)
            agent.reset("s", {})
            first_response = agent.respond(
                "s", "I'm looking for Shoes, but I'm still exploring.", 1, 10
            )
            self.assertEqual(len(first_response["recommendations"]), 1)

            response = agent.respond(
                "s", "For that, what matters is: full grain leather; wide width.", 2, 10
            )
            self.assertEqual(len(response["recommendations"]), 1)
            self.assertEqual(response["recommendations"][0]["parent_asin"], "B")

    def test_cache_capacity_does_not_change_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = self._write_catalog(directory)
            small_cache = Agent(catalog, feature_cache_size=1)
            normal_cache = Agent(catalog, feature_cache_size=5000)
            for agent in (small_cache, normal_cache):
                agent.reset("s", {})
                agent.respond(
                    "s", "I'm looking for Shoes, but I'm still exploring.", 1, 10
                )
            message = "For that, what matters is: full grain leather; wide width."
            small_response = small_cache.respond("s", message, 2, 10)
            normal_response = normal_cache.respond("s", message, 2, 10)
            self.assertEqual(
                small_response["recommendations"], normal_response["recommendations"]
            )


if __name__ == "__main__":
    unittest.main()
