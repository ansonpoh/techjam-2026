from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from starter.dialogue import Evidence, SessionState
from starter.product_features import (
    BUDGET_RE,
    FIELD_WEIGHTS,
    CompiledQuery,
    ProductFeatures,
    ProductFeatureStore,
    ProductQuestionFeatures,
    terms,
)
from starter.ranking import (
    DEFAULT_RANKING_POLICIES,
    IntentRouter,
    RankingMode,
    RankingPolicies,
    RankingPolicy,
)
from starter.vector_index import CatalogVectorIndex, VectorIndex


QUALITY_REVIEW_WEIGHT = 1.05
FEATURE_CACHE_SIZE = 5_000
VECTOR_ROUTE_LIMIT = 250
# Calibrated from docs/vector_gate_calibration.json. These are the 10th
# percentiles for winning target cosine and target-to-runner-up margin.
VECTOR_MIN_SIMILARITY = 0.616618
VECTOR_MIN_MARGIN = 0.011216
# Preserve the old RRF vector route's theoretical maximum: 85 * 0.2 / (60 + 1).
VECTOR_MAX_CONTRIBUTION = 85.0 * 0.2 / 61.0


@dataclass(frozen=True)
class SearchResult:
    recommendations: list[tuple[str, float]]
    candidates: list[dict]
    prompt_tokens: int = 0
    ranking_mode: RankingMode | None = None


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
        *,
        enable_vector_reranker: bool = False,
        ranking_policies: RankingPolicies = DEFAULT_RANKING_POLICIES,
        intent_router: IntentRouter | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.feature_store = ProductFeatureStore(max_size=feature_cache_size)
        self.ranking_policies = ranking_policies
        self.intent_router = intent_router or IntentRouter()
        self._build_index()
        self.vector_index = vector_index
        if self.vector_index is None and enable_vector_reranker:
            self.vector_index = CatalogVectorIndex(self.catalog_path)

    def close(self) -> None:
        if self.vector_index is not None:
            self.vector_index.close()
        self.connection.close()

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

    def _vector_route(self, rows: list[tuple[int, float]]) -> list[dict]:
        if not rows:
            return []
        row_ids = [row_id for row_id, _ in rows]
        placeholders = ",".join("?" for _ in row_ids)
        fetched = self.connection.execute(
            "SELECT rowid, parent_asin, title, categories, features, details, store, "
            "description, price, average_rating, rating_number "
            f"FROM products WHERE rowid IN ({placeholders})",
            row_ids,
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "features", "details", "store",
            "description", "price", "average_rating", "rating_number",
        )
        by_row_id = {
            int(row[0]): dict(zip(keys, row[1:]))
            for row in fetched
        }
        result: list[dict] = []
        for row_id, vector_score in rows:
            product = by_row_id.get(row_id)
            if product is not None:
                product["_vector_score"] = vector_score
                result.append(product)
        return result

    def search(self, state: SessionState, limit: int = 10) -> list[tuple[str, float]]:
        return self.search_with_context(state, limit).recommendations

    def search_with_context(self, state: SessionState, limit: int = 10) -> SearchResult:
        if not state.evidence:
            return SearchResult(recommendations=[], candidates=[])

        routes: list[tuple[float, list[dict]]] = []
        routes.append((1.0, self._route(
            _or_expression([item.text for item in state.evidence]), 350
        )))

        latest = state.latest_evidence
        if latest is not None:
            phrase_route = self._route(_phrase_expression(state.evidence), 180)
            if phrase_route:
                routes.append((1.0, phrase_route))

        if state.category_text:
            category_route = self._route(_or_expression([state.category_text], limit=16), 180)
            if category_route:
                routes.append((1.0, category_route))

        rrf: defaultdict[str, float] = defaultdict(float)
        candidates: dict[str, dict] = {}
        for route_weight, route in routes:
            for rank, product in enumerate(route, start=1):
                parent_asin = str(product["parent_asin"])
                rrf[parent_asin] += route_weight / (60.0 + rank)
                candidates.setdefault(parent_asin, product)

        query = self.feature_store.compile_query(state.evidence, state.user_profile)
        routing = self.intent_router.route(state)
        policy = self.ranking_policies.for_mode(routing.mode)
        base_scores: dict[str, float] = {}
        exact_hard_matches: set[str] = set()
        for parent_asin, product in candidates.items():
            features = product["_features"]
            needs_facets = policy.contradiction_penalty > 0.0 and any(
                item.source in {"hard_constraint", "override"} and item.facets
                for item in query.evidence
            )
            question_features = (
                self.feature_store.question_features(product)
                if needs_facets
                else None
            )
            score = 85.0 * policy.rrf_scale * rrf[parent_asin]
            score += policy.constraint_scale * self._constraint_score(features, query)
            score += policy.price_scale * self._price_score(features, query)
            score += policy.quality_scale * self._quality_tiebreak(features)
            score += self._constraint_fit_adjustment(
                features, question_features, query, policy
            )
            score += self._budget_violation_adjustment(features, query, policy)
            base_scores[parent_asin] = score
            if self._exact_hard_constraint_match(features, state.evidence):
                exact_hard_matches.add(parent_asin)

        # Dense retrieval never admits candidates. It can only adjust lexical
        # candidates after query, category, absolute-similarity, and margin gates.
        vector_prompt_tokens = 0
        vector_scores: dict[str, float] = {}
        vector_confident = False
        structured_query = state.semantic_query()
        if (
            policy.vector_scale > 0.0
            and self.vector_index is not None
            and candidates
            and state.category_text
            and structured_query
        ):
            vector_result = self.vector_index.search(structured_query, VECTOR_ROUTE_LIMIT)
            vector_prompt_tokens = vector_result.prompt_tokens
            category_vector_route = [
                product
                for product in self._vector_route(vector_result.rows)
                if self._category_match(product, state.category_text)
            ]
            if category_vector_route:
                top_score = float(category_vector_route[0]["_vector_score"])
                runner_score = (
                    float(category_vector_route[1]["_vector_score"])
                    if len(category_vector_route) > 1
                    else VECTOR_MIN_SIMILARITY
                )
                top_id = str(category_vector_route[0]["parent_asin"])
                vector_confident = (
                    top_id in candidates
                    and top_score >= VECTOR_MIN_SIMILARITY
                    and top_score - runner_score >= VECTOR_MIN_MARGIN
                )
                vector_scores = {
                    str(product["parent_asin"]): float(product["_vector_score"])
                    for product in category_vector_route
                    if str(product["parent_asin"]) in candidates
                }

        ranked: list[tuple[str, float]] = []
        best_lexical_score = max(base_scores.values(), default=0.0)
        has_exact_hard_match = bool(exact_hard_matches)
        for parent_asin, base_score in base_scores.items():
            similarity = vector_scores.get(parent_asin, 0.0)
            contribution = self._bounded_vector_contribution(
                similarity=similarity,
                base_score=base_score,
                best_lexical_score=best_lexical_score,
                vector_confident=vector_confident,
                has_exact_hard_match=has_exact_hard_match,
                is_exact_hard_match=parent_asin in exact_hard_matches,
            ) * policy.vector_scale
            candidates[parent_asin]["_vector_score"] = similarity
            candidates[parent_asin]["_vector_contribution"] = contribution
            candidates[parent_asin]["_ranking_mode"] = routing.mode.value
            ranked.append((parent_asin, base_score + contribution))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        context: list[dict] = []
        for parent_asin, score in ranked[:100]:
            product = dict(candidates[parent_asin])
            product["_rank_score"] = score
            context.append(product)
        return SearchResult(
            recommendations=ranked[:limit],
            candidates=context,
            prompt_tokens=vector_prompt_tokens,
            ranking_mode=routing.mode,
        )

    @staticmethod
    def _constraint_fit_adjustment(
        product: ProductFeatures,
        product_facets: ProductQuestionFeatures | None,
        query: CompiledQuery,
        policy: RankingPolicy,
    ) -> float:
        score = 0.0
        hard_sources = {"hard_constraint", "override"}
        for item in query.evidence:
            if not item.tokens or item.source == "category" or item.is_budget:
                continue
            matched = sum(token in product.token_weights for token in item.tokens)
            coverage = matched / len(item.tokens)
            exact = (
                len(item.tokens) >= 2
                and item.normalized_query in product.normalized_text
            )
            if item.source in hard_sources:
                score += item.weight * (
                    policy.hard_coverage_bonus * coverage
                    - policy.hard_missing_penalty * (1.0 - coverage)
                    + policy.hard_exact_bonus * float(exact)
                )
                for attribute, expected_values in item.facets:
                    actual_values = set(
                        product_facets.facet_values(attribute)
                        if product_facets is not None
                        else ()
                    )
                    if (
                        expected_values
                        and actual_values
                        and actual_values.isdisjoint(expected_values)
                    ):
                        score -= item.weight * policy.contradiction_penalty
            else:
                score += item.weight * (
                    policy.soft_coverage_bonus * coverage
                    + policy.soft_exact_bonus * float(exact)
                )
        return score

    @staticmethod
    def _budget_violation_adjustment(
        product: ProductFeatures,
        query: CompiledQuery,
        policy: RankingPolicy,
    ) -> float:
        if product.price is None or policy.budget_violation_penalty <= 0.0:
            return 0.0
        score = 0.0
        for budget in query.budgets:
            relative_error = abs(product.price - budget.amount) / max(
                budget.amount, 10.0
            )
            if budget.mode in {"under", "below", "maximum", "max"}:
                violation = max(0.0, (product.price - budget.amount) / budget.amount)
            else:
                violation = max(0.0, relative_error - 0.35)
            score -= (
                budget.weight
                * policy.budget_violation_penalty
                * min(violation, 2.0)
            )
        return score

    @staticmethod
    def _category_match(product: dict, requested_category: str) -> bool:
        requested = set(terms(requested_category))
        product_categories = set(terms(str(product.get("categories") or "")))
        return bool(requested) and requested.issubset(product_categories)

    @staticmethod
    def _bounded_vector_contribution(
        *,
        similarity: float,
        base_score: float,
        best_lexical_score: float,
        vector_confident: bool,
        has_exact_hard_match: bool,
        is_exact_hard_match: bool,
    ) -> float:
        if (
            not vector_confident
            or similarity < VECTOR_MIN_SIMILARITY
            or best_lexical_score - base_score > VECTOR_MAX_CONTRIBUTION
            or (has_exact_hard_match and not is_exact_hard_match)
        ):
            return 0.0
        return min(
            VECTOR_MAX_CONTRIBUTION,
            max(0.0, similarity) * VECTOR_MAX_CONTRIBUTION,
        )

    @staticmethod
    def _exact_hard_constraint_match(
        product: ProductFeatures | dict, evidence: list[Evidence]
    ) -> bool:
        hard_constraints = [
            item
            for item in evidence
            if item.source in {"hard_constraint", "override"}
            and not BUDGET_RE.search(item.text)
            and terms(item.text)
        ]
        if not hard_constraints:
            return False
        if isinstance(product, ProductFeatures):
            normalized_text = product.normalized_text
        else:
            normalized_text = "\x1f".join(
                " ".join(terms(str(product.get(field) or "")))
                for field in FIELD_WEIGHTS
            )
        return all(
            " ".join(terms(item.text)) in normalized_text
            for item in hard_constraints
        )

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
