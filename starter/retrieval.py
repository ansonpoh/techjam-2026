from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from starter.dialogue import Evidence, SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
BUDGET_RE = re.compile(
    r"\b(?P<mode>under|below|maximum|max|around|about|budget(?:\s+around)?)?\s*"
    r"\$\s*(?P<amount>\d+(?:\.\d+)?)",
    re.I,
)
STOPWORDS = {
    "a", "about", "additional", "am", "an", "and", "are", "as", "at", "be",
    "but", "by", "do", "for", "from", "have", "i", "in", "is", "it", "looking",
    "me", "my", "need", "not", "of", "on", "or", "please", "preference", "some",
    "still", "that", "the", "these", "this", "those", "to", "want", "what", "with",
    "would", "you", "your",
}
FIELD_WEIGHTS = {
    "title": 4.0,
    "categories": 3.0,
    "features": 2.8,
    "details": 2.8,
    "store": 1.5,
    "description": 1.3,
}
QUALITY_REVIEW_WEIGHT = 1.05


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


def terms(value: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _or_expression(values: list[str], limit: int = 48) -> str:
    unique = list(dict.fromkeys(token for value in values for token in terms(value)))[:limit]
    return " OR ".join(f'"{token}"' for token in unique)


def _phrase_expression(evidence: list[Evidence], limit: int = 4) -> str:
    chunks = sorted(
        (item for item in evidence if item.source != "category" and terms(item.text)),
        key=lambda item: (len(set(terms(item.text))), item.weight, item.turn),
        reverse=True,
    )
    phrases: list[str] = []
    for item in chunks[:limit]:
        chunk_terms = terms(item.text)[:14]
        if chunk_terms:
            phrases.append('"' + " ".join(chunk_terms) + '"')
    return " OR ".join(phrases)


class CatalogSearch:
    """Multi-route FTS retrieval plus deterministic constraint reranking."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
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
                batch.append((
                    str(product["parent_asin"]),
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
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
        return [dict(zip(keys, row)) for row in rows]

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

        ranked: list[tuple[str, float]] = []
        for parent_asin, product in candidates.items():
            score = 85.0 * rrf[parent_asin]
            score += self._constraint_score(product, state.evidence, state.user_profile)
            score += self._price_score(product, state.evidence)
            score += self._quality_tiebreak(product)
            ranked.append((parent_asin, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        context: list[dict] = []
        for parent_asin, score in ranked[:100]:
            product = dict(candidates[parent_asin])
            product["_rank_score"] = score
            context.append(product)
        return SearchResult(recommendations=ranked[:limit], candidates=context)

    @staticmethod
    def _constraint_score(
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
                specificity = min(2.0, 0.55 + 0.22 * len(query_terms))
                score += item.weight * specificity
            if coverage >= 0.999:
                score += item.weight * 0.45
        tags = user_profile.get("preference_tags") if isinstance(user_profile, dict) else None
        if isinstance(tags, list) and tags:
            preference_terms = {token for tag in tags for token in terms(str(tag))}
            product_terms = set().union(*field_tokens.values())
            if preference_terms:
                score += 0.45 * len(preference_terms & product_terms) / len(preference_terms)
        return score

    @staticmethod
    def _price_score(product: dict, evidence: list[Evidence]) -> float:
        try:
            price = float(product.get("price") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if price <= 0.0:
            return 0.0

        score = 0.0
        for item in evidence:
            match = BUDGET_RE.search(item.text)
            if not match:
                continue
            amount = float(match.group("amount"))
            mode = (match.group("mode") or "around").lower()
            if mode in {"under", "below", "maximum", "max"}:
                closeness = 1.0 if price <= amount else max(0.0, 1.0 - (price - amount) / amount)
            else:
                closeness = max(0.0, 1.0 - abs(price - amount) / max(amount, 10.0))
            score += item.weight * 1.4 * closeness
        return score

    @staticmethod
    def _quality_tiebreak(product: dict) -> float:
        try:
            rating = float(product.get("average_rating") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        try:
            count = max(0, int(float(product.get("rating_number") or 0)))
        except (TypeError, ValueError):
            count = 0
        return (
            min(max(rating, 0.0), 5.0) * 0.02
            + math.log1p(count) * QUALITY_REVIEW_WEIGHT
        )
