from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from starter.dialogue import Evidence, SessionState
from starter.product_features import (
    FIELD_WEIGHTS,
    CompiledQuery,
    ProductFeatures,
    ProductFeatureStore,
    terms,
)


QUALITY_REVIEW_WEIGHT = 1.05
FEATURE_CACHE_SIZE = 5_000


@dataclass(frozen=True)
class SearchResult:
    recommendations: list[tuple[str, float]]
    candidates: list[dict]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _or_expression(values: list[str], limit: int = 48) -> str:
    unique = list(dict.fromkeys(token for value in values for token in terms(value)))[:limit]
    return " OR ".join(f'"{token}"' for token in unique)


def _phrase_expression(evidence: list[Evidence], limit: int = 4) -> str:
    tokenized = [
        (item, terms(item.text))
        for item in evidence
        if item.source != "category"
    ]
    chunks = sorted(
        ((item, item_terms) for item, item_terms in tokenized if item_terms),
        key=lambda pair: (len(set(pair[1])), pair[0].weight, pair[0].turn),
        reverse=True,
    )
    phrases: list[str] = []
    for _, item_terms in chunks[:limit]:
        chunk_terms = item_terms[:14]
        if chunk_terms:
            phrases.append('"' + " ".join(chunk_terms) + '"')
    return " OR ".join(phrases)


class CatalogSearch:
    """Multi-route FTS retrieval plus deterministic constraint reranking."""

    def __init__(
        self,
        catalog_path: str | Path,
        feature_cache_size: int = FEATURE_CACHE_SIZE,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.feature_store = ProductFeatureStore(max_size=feature_cache_size)
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                fields = {
                    "title": _text(product.get("title")),
                    "categories": _text(product.get("categories")),
                    "features": _text(product.get("features")),
                    "details": _text(product.get("details")),
                    "store": _text(product.get("store")),
                    "description": _text(product.get("description")),
                }
                batch.append((
                    parent_asin,
                    fields["title"],
                    fields["categories"],
                    fields["features"],
                    fields["details"],
                    fields["store"],
                    fields["description"],
                    _text(product.get("price")),
                    _text(product.get("average_rating")),
                    _text(product.get("rating_number")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
        self.connection.commit()

    def _route(self, expression: str, limit: int) -> list[dict]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description, "
            "price, average_rating, rating_number "
            "FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 7.0, 4.5, 3.2, 3.2, 1.8, 1.2, 0.0, 0.0, 0.0) "
            "LIMIT ?",
            (expression, limit),
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "features", "details", "store",
            "description", "price", "average_rating", "rating_number",
        )
        products: list[dict] = []
        for row in rows:
            product = dict(zip(keys, row))
            fields = {
                field: str(product.get(field) or "")
                for field in FIELD_WEIGHTS
            }
            product["_features"] = self.feature_store.get_or_add(
                str(product["parent_asin"]),
                fields,
                price=product.get("price"),
                average_rating=product.get("average_rating"),
                rating_number=product.get("rating_number"),
            )
            products.append(product)
        return products

    def search(self, state: SessionState, limit: int = 10) -> list[tuple[str, float]]:
        return self.search_with_context(state, limit).recommendations

    def search_with_context(self, state: SessionState, limit: int = 10) -> SearchResult:
        if not state.evidence:
            return SearchResult(recommendations=[], candidates=[])

        routes: list[list[dict]] = []
        routes.append(self._route(_or_expression([item.text for item in state.evidence]), 350))

        latest = state.latest_evidence
        if latest is not None:
            phrase_route = self._route(_phrase_expression(state.evidence), 180)
            if phrase_route:
                routes.append(phrase_route)

        if state.category_text:
            category_route = self._route(_or_expression([state.category_text], limit=16), 180)
            if category_route:
                routes.append(category_route)

        rrf: defaultdict[str, float] = defaultdict(float)
        candidates: dict[str, dict] = {}
        for route in routes:
            for rank, product in enumerate(route, start=1):
                parent_asin = str(product["parent_asin"])
                rrf[parent_asin] += 1.0 / (60.0 + rank)
                candidates.setdefault(parent_asin, product)

        query = self.feature_store.compile_query(state.evidence, state.user_profile)
        ranked: list[tuple[str, float]] = []
        for parent_asin, product in candidates.items():
            features = product["_features"]
            score = 85.0 * rrf[parent_asin]
            score += self._constraint_score(features, query)
            score += self._price_score(features, query)
            score += self._quality_tiebreak(features)
            ranked.append((parent_asin, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        context: list[dict] = []
        for parent_asin, score in ranked[:100]:
            product = dict(candidates[parent_asin])
            product["_rank_score"] = score
            context.append(product)
        return SearchResult(recommendations=ranked[:limit], candidates=context)

    @staticmethod
    def _constraint_score(product: ProductFeatures, query: CompiledQuery) -> float:
        score = 0.0
        for item in query.evidence:
            if not item.tokens:
                continue
            matched_weight = 0.0
            matched_terms = 0
            for token in item.tokens:
                best_field_weight = product.token_weights.get(token, 0.0)
                matched_weight += best_field_weight
                matched_terms += int(best_field_weight > 0.0)
            coverage = matched_terms / len(item.tokens)
            field_affinity = matched_weight / (
                len(item.tokens) * max(FIELD_WEIGHTS.values())
            )
            score += item.weight * (1.9 * coverage + 0.4 * field_affinity)

            if len(item.tokens) >= 2 and item.normalized_query in product.normalized_text:
                specificity = min(2.0, 0.55 + 0.22 * len(item.tokens))
                score += item.weight * specificity
            if coverage >= 0.999:
                score += item.weight * 0.45
        if query.preference_tokens:
            matches = sum(
                token in product.token_weights
                for token in query.preference_tokens
            )
            score += 0.45 * matches / len(query.preference_tokens)
        return score

    @staticmethod
    def _price_score(product: ProductFeatures, query: CompiledQuery) -> float:
        if product.price is None:
            return 0.0

        score = 0.0
        for budget in query.budgets:
            if budget.mode in {"under", "below", "maximum", "max"}:
                closeness = (
                    1.0
                    if product.price <= budget.amount
                    else max(
                        0.0,
                        1.0 - (product.price - budget.amount) / budget.amount,
                    )
                )
            else:
                closeness = max(
                    0.0,
                    1.0
                    - abs(product.price - budget.amount) / max(budget.amount, 10.0),
                )
            score += budget.weight * 1.4 * closeness
        return score

    @staticmethod
    def _quality_tiebreak(product: ProductFeatures) -> float:
        return (
            min(max(product.average_rating, 0.0), 5.0) * 0.02
            + math.log1p(product.rating_number) * QUALITY_REVIEW_WEIGHT
        )
