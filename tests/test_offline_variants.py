from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.dialogue import Evidence, SessionState
from starter.offline_variants import OfflineVariantMatcher, character_ngrams
from starter.retrieval import CatalogSearch


class OfflineVariantMatcherTest(unittest.TestCase):
    def test_character_ngrams_include_word_boundaries(self) -> None:
        self.assertEqual(character_ngrams("shoe"), {"^sh", "sho", "hoe", "oe$"})

    def test_curated_synonym_normalization_is_one_way(self) -> None:
        matcher = OfflineVariantMatcher(vocabulary=[("shoes", 100)])

        rewrite = matcher.rewrite(["comfortable trainers"])

        self.assertIn("sneakers", rewrite.synonym_tokens)
        self.assertIn("shoes", rewrite.expanded_tokens)
        self.assertTrue(rewrite.changed)

        known_variant = OfflineVariantMatcher(vocabulary=[("tees", 100)])
        self.assertFalse(known_variant.rewrite(["tees"]).changed)

    def test_high_confidence_typo_is_corrected_at_token_level(self) -> None:
        matcher = OfflineVariantMatcher(vocabulary=[
            ("sneakers", 100), ("speakers", 80), ("sweaters", 60),
        ])

        rewrite = matcher.rewrite(["snekaers"])

        self.assertEqual(rewrite.fuzzy_tokens, (("snekaers", "sneakers"),))

    def test_known_and_ambiguous_tokens_are_not_rewritten(self) -> None:
        matcher = OfflineVariantMatcher(vocabulary=[
            ("boots", 100), ("coats", 10), ("boats", 10),
        ])

        self.assertFalse(matcher.rewrite(["boots"]).changed)
        self.assertFalse(matcher.rewrite(["boets"]).changed)


class OfflineVariantRetrievalTest(unittest.TestCase):
    def test_typo_route_recovers_catalog_candidate_without_network(self) -> None:
        rows = [
            {
                "parent_asin": "TARGET", "title": "Trail Sneakers",
                "categories": ["Shoes"], "features": ["breathable mesh"],
                "details": {}, "store": "Example", "description": [],
                "price": 50, "average_rating": 4.5, "rating_number": 100,
            },
            {
                "parent_asin": "OTHER", "title": "Formal Shoes",
                "categories": ["Shoes"], "features": ["leather"],
                "details": {}, "store": "Example", "description": [],
                "price": 50, "average_rating": 4.5, "rating_number": 100,
            },
            {
                "parent_asin": "OUTSIDE", "title": "Sneakers Graphic Tee",
                "categories": ["Shirts"], "features": ["cotton"],
                "details": {}, "store": "Example", "description": [],
                "price": 20, "average_rating": 4.0, "rating_number": 10,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            search = CatalogSearch(catalog, use_prebuilt_index=False)
            state = SessionState(user_profile={}, category_text="Shoes")
            state.evidence.extend([
                Evidence("Shoes", 1.4, "category", 1),
                Evidence("snekaers", 3.8, "hard_constraint", 1),
            ])
            try:
                result = search.search_with_context(state, limit=2)
            finally:
                search.close()

        self.assertEqual(result.recommendations[0][0], "TARGET")
        self.assertEqual(
            result.candidates[0]["_offline_variant_rewrite"]["fuzzy"],
            (("snekaers", "sneakers"),),
        )


if __name__ == "__main__":
    unittest.main()
