from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from starter.dialogue import Evidence, SessionState
from starter.ranking import RankingMode
from starter.retrieval import (
    CatalogSearch,
    VECTOR_MAX_CONTRIBUTION,
    VECTOR_MIN_MARGIN,
    VECTOR_MIN_SIMILARITY,
)
from starter.vector_index import CatalogVectorIndex, VectorSearchResult, catalog_sha256
from scripts.generate_catalog_embeddings import generate


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]], prompt_tokens: int = 7) -> None:
        self.vectors = vectors
        self.prompt_tokens = prompt_tokens
        self.calls: list[list[str]] = []

    def create(self, **kwargs: object) -> object:
        inputs = list(kwargs["input"])
        self.calls.append(inputs)
        data = [
            SimpleNamespace(index=index, embedding=self.vectors[text])
            for index, text in enumerate(inputs)
        ]
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(prompt_tokens=self.prompt_tokens),
        )


class FakeClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.embeddings = FakeEmbeddings(vectors)


class FailingEmbeddings:
    def create(self, **kwargs: object) -> object:
        raise RuntimeError("simulated outage")


class FailingClient:
    embeddings = FailingEmbeddings()


class BatchEmbeddings:
    def __init__(self, interrupt_on_call: int | None = None) -> None:
        self.calls = 0
        self.interrupt_on_call = interrupt_on_call

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        inputs = list(kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[1.0, 0.0])
                for index, _ in enumerate(inputs)
            ],
            usage=SimpleNamespace(prompt_tokens=len(inputs)),
        )


class BatchClient:
    def __init__(self, interrupt_on_call: int | None = None) -> None:
        self.embeddings = BatchEmbeddings(interrupt_on_call)


class StubVectorIndex:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self.rows = rows
        self.calls: list[str | None] = []

    def search(self, query: str | None, limit: int) -> VectorSearchResult:
        self.calls.append(query)
        return VectorSearchResult(rows=self.rows[:limit])

    def close(self) -> None:
        pass


def write_catalog(path: Path) -> None:
    rows = [
        {
            "parent_asin": "A", "title": "Red City Shoe", "categories": ["Shoes"],
            "features": ["lightweight"], "details": {}, "store": "Example",
            "description": [], "price": 40, "average_rating": 4.0, "rating_number": 10,
        },
        {
            "parent_asin": "B", "title": "Leather Trail Boot", "categories": ["Shoes"],
            "features": ["wide width"], "details": {}, "store": "Example",
            "description": [], "price": 60, "average_rating": 4.0, "rating_number": 10,
        },
        {
            "parent_asin": "C", "title": "Blue Summer Shirt", "categories": ["Shirts"],
            "features": ["cotton"], "details": {}, "store": "Example",
            "description": [], "price": 20, "average_rating": 4.0, "rating_number": 10,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_artifact(directory: Path, catalog: Path, *, checksum: str | None = None) -> tuple[Path, Path]:
    vectors_path = directory / "catalog_embeddings.npy"
    metadata_path = directory / "catalog_embeddings.meta.json"
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.70710677, 0.70710677]], dtype=np.float32)
    np.save(vectors_path, vectors)
    metadata_path.write_text(
        json.dumps({
            "model": "fake-embedding-model",
            "dimensions": 2,
            "row_count": 3,
            "catalog_sha256": checksum or catalog_sha256(catalog),
            "normalized": True,
        }),
        encoding="utf-8",
    )
    return vectors_path, metadata_path


class CatalogVectorIndexTest(unittest.TestCase):
    def test_catalog_checksum_is_independent_of_git_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            lf_catalog = directory / "catalog-lf.jsonl"
            crlf_catalog = directory / "catalog-crlf.jsonl"
            content = b'{"parent_asin":"A"}\n{"parent_asin":"B"}\n'
            lf_catalog.write_bytes(content)
            crlf_catalog.write_bytes(content.replace(b"\n", b"\r\n"))

            self.assertEqual(
                catalog_sha256(lf_catalog),
                catalog_sha256(crlf_catalog),
            )

    def test_exact_similarity_and_cache_avoid_second_api_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            query = "Required features: red city shoe"
            client = FakeClient({query: [1.0, 0.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            first = index.search(query, limit=2)
            second = index.search(query, limit=2)

            self.assertEqual(first.rows[0][0], 1)
            self.assertEqual(first.prompt_tokens, 7)
            self.assertEqual(second.prompt_tokens, 0)
            self.assertEqual(len(client.embeddings.calls), 1)
            index.close()

    def test_structured_query_is_embedded_once_without_vector_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            query = "Product category: Shoes\nRequired features: wide width\nIntended use: trail"
            client = FakeClient({query: [0.0, 1.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )

            result = index.search(query, limit=1)

            self.assertEqual(result.rows[0][0], 2)
            self.assertEqual(client.embeddings.calls, [[query]])
            index.close()

    def test_catalog_checksum_mismatch_disables_vector_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(
                directory, catalog, checksum=hashlib.sha256(b"wrong").hexdigest()
            )
            client = FakeClient({"shoe": [1.0, 0.0]})

            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )

            self.assertFalse(index.enabled)
            self.assertEqual(index.search("Product category: Shoes").rows, [])
            self.assertEqual(client.embeddings.calls, [])
            index.close()

    def test_vector_route_only_reranks_lexical_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            query = "Product category: Shoes\nRequired features: wide width"
            client = FakeClient({query: [0.0, 1.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
            state.record_question("other")
            state.observe("For that, what matters is: wide width.", 2)

            result = search.search_with_context(state, limit=3)

            recommendation_ids = {parent_asin for parent_asin, _ in result.recommendations}
            self.assertEqual(recommendation_ids, {"A", "B"})
            self.assertNotIn("C", recommendation_ids)
            self.assertEqual(result.prompt_tokens, 7)
            search.close()

    def test_similarity_margin_gate_disables_ambiguous_vector_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            catalog = Path(directory_name) / "catalog.jsonl"
            write_catalog(catalog)
            vector_index = StubVectorIndex([(1, 0.700), (2, 0.695), (3, 0.100)])
            search = CatalogSearch(catalog, vector_index=vector_index)
            state = SessionState(user_profile={})
            state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
            state.record_question("other")
            state.observe("For that, what matters is: shoe boot.", 2)

            result = search.search_with_context(state, limit=3)

            self.assertEqual(len(vector_index.calls), 1)
            self.assertTrue(result.candidates)
            self.assertTrue(all(
                product["_vector_contribution"] == 0.0
                for product in result.candidates
            ))
            search.close()

    def test_vector_gates_and_contribution_are_bounded(self) -> None:
        common = {
            "base_score": 10.0,
            "best_lexical_score": 10.0,
            "vector_confident": True,
            "has_exact_hard_match": False,
            "is_exact_hard_match": False,
        }
        lower = CatalogSearch._bounded_vector_contribution(similarity=0.70, **common)
        higher = CatalogSearch._bounded_vector_contribution(similarity=0.90, **common)
        self.assertGreater(higher, lower)
        self.assertLessEqual(higher, VECTOR_MAX_CONTRIBUTION)
        self.assertEqual(CatalogSearch._bounded_vector_contribution(
            similarity=VECTOR_MIN_SIMILARITY - 0.001, **common
        ), 0.0)
        self.assertEqual(CatalogSearch._bounded_vector_contribution(
            similarity=0.90,
            **{**common, "base_score": 10.0 - VECTOR_MAX_CONTRIBUTION - 0.001},
        ), 0.0)
        self.assertEqual(CatalogSearch._bounded_vector_contribution(
            similarity=0.90,
            **{**common, "has_exact_hard_match": True},
        ), 0.0)

    def test_category_and_exact_hard_constraint_guards(self) -> None:
        shoe = {"categories": "Clothing Shoes", "features": "wide width"}
        shirt = {"categories": "Clothing Shirts", "features": "wide width"}
        hard = [Evidence("wide width", 3.8, "hard_constraint", 1)]

        self.assertTrue(CatalogSearch._category_match(shoe, "Shoes"))
        self.assertFalse(CatalogSearch._category_match(shirt, "Shoes"))
        self.assertTrue(CatalogSearch._exact_hard_constraint_match(shoe, hard))
        self.assertFalse(CatalogSearch._exact_hard_constraint_match(
            {**shoe, "features": "standard width"}, hard
        ))

    def test_runtime_thresholds_match_calibration_artifact(self) -> None:
        calibration = json.loads(
            (Path(__file__).resolve().parents[1] / "docs" / "vector_gate_calibration.json")
            .read_text(encoding="utf-8")
        )
        selected = calibration["selected_thresholds"]
        self.assertEqual(VECTOR_MIN_SIMILARITY, selected["minimum_cosine_similarity"])
        self.assertEqual(VECTOR_MIN_MARGIN, selected["minimum_top_margin"])

    def test_empty_lexical_pool_skips_vector_candidate_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({"Required features: mountain footwear": [0.0, 1.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.evidence.append(Evidence("mountain footwear", 3.0, "clarification", 1))

            result = search.search_with_context(state, limit=3)

            self.assertEqual(result.recommendations, [])
            self.assertEqual(result.prompt_tokens, 0)
            self.assertEqual(client.embeddings.calls, [])
            search.close()

    def test_api_failure_falls_back_to_fts_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=FailingClient(),
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
            state.record_question("other")
            state.observe("For that, what matters is: leather trail boot.", 2)

            result = search.search_with_context(state, limit=3)

            self.assertEqual(result.recommendations[0][0], "B")
            self.assertFalse(index.enabled)
            search.close()

    def test_intent_override_excludes_superseded_evidence_from_embedding(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes. I prefer red.", 1)
        state.observe("Actually, ignore my earlier preference. What I need is: trail.", 2)
        self.assertNotIn("i prefer red", [item.text.casefold() for item in state.evidence])
        self.assertIn("trail", [item.text.casefold() for item in state.evidence])
        self.assertEqual(
            state.semantic_query(),
            "Product category: Shoes\nRequired features: trail",
        )

    def test_semantic_query_excludes_generic_and_repeated_evidence(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        self.assertIsNone(state.semantic_query())

        for category in ("Shoes", "Jewelry"):
            with self.subTest(category=category):
                generic_state = SessionState(user_profile={})
                generic_state.observe(category, 1)
                self.assertIsNone(generic_state.semantic_query())

        state.record_question("other")
        state.observe("For that, what matters is: Shoes; waterproof; wide width.", 2)
        self.assertEqual(
            state.semantic_query(),
            "Product category: Shoes\nRequired features: waterproof; wide width",
        )

    def test_semantic_query_labels_use_case_and_skips_no_preference(self) -> None:
        state = SessionState(user_profile={})
        state.observe("I'm looking for Shoes, but I'm still exploring.", 1)
        state.record_question("use_case")
        state.observe("For that, what matters is: trail running.", 2)
        state.record_question("color")
        state.observe("I don't have a preference for color; please use your judgment.", 3)

        self.assertEqual(
            state.semantic_query(),
            "Product category: Shoes\nIntended use: trail running",
        )

    def test_generic_category_does_not_call_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.observe("I'm looking for Shoes, but I'm still exploring.", 1)

            result = search.search_with_context(state, limit=3)

            self.assertEqual(result.prompt_tokens, 0)
            self.assertEqual(client.embeddings.calls, [])
            search.close()

    def test_buying_mode_skips_vector_api_and_uses_precision_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.observe(
                "I'm looking for Shoes. A key requirement is: wide width.", 1
            )

            result = search.search_with_context(state, limit=3)

            self.assertEqual(result.ranking_mode, RankingMode.BUYING)
            self.assertEqual(result.prompt_tokens, 0)
            self.assertEqual(client.embeddings.calls, [])
            search.close()

    def test_catalog_generator_resumes_after_completed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            output = directory / "vectors.npy"
            metadata = directory / "vectors.meta.json"
            args = Namespace(
                catalog=str(catalog), output=str(output), metadata=str(metadata),
                model="fake-embedding-model", dimensions=2, batch_size=2,
            )

            with (
                patch("scripts.generate_catalog_embeddings.load_openai_api_key", return_value=True),
                patch(
                    "scripts.generate_catalog_embeddings.create_openai_client",
                    return_value=BatchClient(interrupt_on_call=2),
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                generate(args)

            progress = json.loads(
                (directory / "vectors.npy.progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["completed_rows"], 2)

            with (
                patch("scripts.generate_catalog_embeddings.load_openai_api_key", return_value=True),
                patch(
                    "scripts.generate_catalog_embeddings.create_openai_client",
                    return_value=BatchClient(),
                ),
            ):
                generate(args)

            self.assertEqual(np.load(output, allow_pickle=False).shape, (3, 2))
            final_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(final_metadata["row_count"], 3)
            self.assertFalse((directory / "vectors.npy.progress.json").exists())


if __name__ == "__main__":
    unittest.main()
