from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.dialogue import Evidence, SessionState
from starter.product_features import FIELD_WEIGHTS, ProductFeatureStore, terms
from starter.question_planner import AdaptiveQuestionPlanner
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
