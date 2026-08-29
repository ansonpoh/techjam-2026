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
from starter.retrieval import CatalogSearch
from starter.vector_index import CatalogVectorIndex, catalog_sha256
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
    def test_exact_similarity_and_cache_avoid_second_api_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({"red city shoe": [1.0, 0.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            evidence = [Evidence("Red city shoe", 2.0, "clarification", 1)]

            first = index.search(evidence, limit=2)
            second = index.search(evidence, limit=2)

            self.assertEqual(first.rows[0][0], 1)
            self.assertEqual(first.prompt_tokens, 7)
            self.assertEqual(second.prompt_tokens, 0)
            self.assertEqual(len(client.embeddings.calls), 1)
            index.close()

    def test_evidence_weights_change_the_combined_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({"city": [1.0, 0.0], "trail": [0.0, 1.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )

            result = index.search([
                Evidence("city", 1.0, "category", 1),
                Evidence("trail", 4.0, "hard_constraint", 2),
            ], limit=1)

            self.assertEqual(result.rows[0][0], 2)
            self.assertEqual(client.embeddings.calls, [["city", "trail"]])
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
            self.assertEqual(index.search([Evidence("shoe", 1.0, "category", 1)]).rows, [])
            self.assertEqual(client.embeddings.calls, [])
            index.close()

    def test_vector_route_only_reranks_lexical_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({"shoes": [0.0, 1.0]})
            index = CatalogVectorIndex(
                catalog,
                vectors_path=vectors_path,
                metadata_path=metadata_path,
                client=client,
            )
            search = CatalogSearch(catalog, vector_index=index)
            state = SessionState(user_profile={})
            state.evidence.append(Evidence("shoes", 3.0, "clarification", 1))

            result = search.search_with_context(state, limit=3)

            recommendation_ids = {parent_asin for parent_asin, _ in result.recommendations}
            self.assertEqual(recommendation_ids, {"A", "B"})
            self.assertNotIn("C", recommendation_ids)
            self.assertEqual(result.prompt_tokens, 7)
            search.close()

    def test_empty_lexical_pool_skips_vector_candidate_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            catalog = directory / "catalog.jsonl"
            write_catalog(catalog)
            vectors_path, metadata_path = write_artifact(directory, catalog)
            client = FakeClient({"mountain footwear": [0.0, 1.0]})
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
            state.evidence.append(Evidence("leather trail boot", 3.0, "clarification", 1))

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
