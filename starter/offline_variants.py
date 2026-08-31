from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from starter.product_features import terms


# Deliberately one-way: common catalog wording is not broadened, while a small
# set of shopper phrasings is mapped to likely catalog vocabulary.
CURATED_SYNONYMS: dict[str, tuple[str, ...]] = {
    "kicks": ("shoes", "sneakers"),
    "trainers": ("sneakers", "athletic", "shoes"),
    "sneaker": ("sneakers",),
    "tee": ("shirt", "tshirt"),
    "tees": ("shirts", "tshirts"),
    "tshirt": ("shirt",),
    "trousers": ("pants",),
    "slacks": ("pants",),
    "pullover": ("sweater",),
    "jumper": ("sweater",),
    "handbag": ("purse", "bag"),
    "handbags": ("purses", "bags"),
    "flipflops": ("flip", "flops", "sandals"),
    "waterproofed": ("waterproof",),
}

ASCII_WORD_RE = re.compile(r"^[a-z]+$")


def character_ngrams(token: str, size: int = 3) -> frozenset[str]:
    """Return boundary-aware character n-grams for typo candidate recall."""
    padded = f"^{token.casefold()}$"
    if len(padded) <= size:
        return frozenset({padded})
    return frozenset(
        padded[index : index + size]
        for index in range(len(padded) - size + 1)
    )


def _edit_distance(left: str, right: str, maximum: int) -> int:
    """Bounded Levenshtein distance; exits once a match cannot pass the gate."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        row_minimum = row
        for column, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(left_char != right_char),
            ))
            row_minimum = min(row_minimum, current[-1])
        if row_minimum > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class VariantRewrite:
    original_tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    synonym_tokens: tuple[str, ...]
    fuzzy_tokens: tuple[tuple[str, str], ...]

    @property
    def changed(self) -> bool:
        return bool(self.synonym_tokens or self.fuzzy_tokens)


class OfflineVariantMatcher:
    """Catalog-derived, network-free synonym and conservative typo matcher."""

    def __init__(
        self,
        connection: sqlite3.Connection | None = None,
        *,
        vocabulary: Iterable[tuple[str, int]] | None = None,
    ) -> None:
        self.connection = connection
        self._provided_vocabulary = list(vocabulary) if vocabulary is not None else None
        self._vocab_table_ready = False
        self._vocabulary: list[tuple[str, int]] | None = None
        self._vocabulary_set: set[str] | None = None
        self._ngram_index: dict[str, tuple[int, ...]] | None = None
        self._known_cache: dict[str, bool] = {}

    def rewrite(self, values: Iterable[str], limit: int = 48) -> VariantRewrite:
        original = tuple(dict.fromkeys(
            token for value in values for token in terms(value)
        ))
        expanded = list(original)
        synonyms: list[str] = []
        fuzzy: list[tuple[str, str]] = []

        for token in original:
            if self._contains(token):
                continue
            for replacement in CURATED_SYNONYMS.get(token, ()):
                if replacement not in expanded:
                    expanded.append(replacement)
                    synonyms.append(replacement)

        for token in original:
            correction = self._fuzzy_correction(token)
            if correction is not None and correction not in expanded:
                expanded.append(correction)
                fuzzy.append((token, correction))

        return VariantRewrite(
            original_tokens=original,
            expanded_tokens=tuple(expanded[:limit]),
            synonym_tokens=tuple(synonyms),
            fuzzy_tokens=tuple(fuzzy),
        )

    def _fuzzy_correction(self, token: str) -> str | None:
        # Short tokens, numeric measurements, and known catalog terms are too
        # ambiguous to rewrite safely.
        if (
            token in CURATED_SYNONYMS
            or len(token) < 4
            or len(token) > 24
            or not ASCII_WORD_RE.fullmatch(token)
        ):
            return None
        if self._contains(token):
            return None

        self._ensure_ngram_index()
        assert self._vocabulary is not None
        assert self._ngram_index is not None
        query_grams = character_ngrams(token)
        candidate_counts: Counter[int] = Counter()
        for gram in query_grams:
            candidate_counts.update(self._ngram_index.get(gram, ()))
        if not candidate_counts:
            return None

        scores: list[tuple[float, int, int, str]] = []
        for index, shared in candidate_counts.most_common(96):
            candidate, document_frequency = self._vocabulary[index]
            maximum_edits = 1 if max(len(token), len(candidate)) <= 7 else 2
            distance = _edit_distance(token, candidate, maximum_edits)
            if distance > maximum_edits:
                continue
            candidate_grams = character_ngrams(candidate)
            union_size = len(query_grams | candidate_grams)
            ngram_similarity = shared / union_size if union_size else 0.0
            token_similarity = SequenceMatcher(None, token, candidate).ratio()
            score = 0.55 * token_similarity + 0.45 * ngram_similarity
            if token_similarity >= 0.78 and ngram_similarity >= 0.30:
                scores.append((score, -distance, document_frequency, candidate))

        if not scores:
            return None
        scores.sort(reverse=True)
        best = scores[0]
        runner_score = scores[1][0] if len(scores) > 1 else 0.0
        # Require either a clear winner or a very common one-edit catalog term.
        # A transposition changes several character trigrams despite being an
        # obvious token-level match, so the aggregate floor stays below the
        # individual token-similarity gate above.
        if best[0] < 0.62:
            return None
        clear_margin = best[0] - runner_score >= 0.06
        common_one_edit = best[1] == -1 and best[2] >= 5 and best[0] >= 0.80
        return best[3] if clear_margin or common_one_edit else None

    def _contains(self, token: str) -> bool:
        cached = self._known_cache.get(token)
        if cached is not None:
            return cached
        if self._provided_vocabulary is not None:
            if self._vocabulary_set is None:
                self._vocabulary_set = {term for term, _ in self._provided_vocabulary}
            result = token in self._vocabulary_set
            self._known_cache[token] = result
            return result
        if self.connection is None:
            return False
        self._ensure_vocab_table()
        result = self.connection.execute(
            "SELECT 1 FROM temp.product_vocab WHERE term = ? LIMIT 1", (token,)
        ).fetchone() is not None
        self._known_cache[token] = result
        return result

    def _ensure_vocab_table(self) -> None:
        if self._vocab_table_ready:
            return
        if self.connection is None:
            raise RuntimeError("catalog connection is required")
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.product_vocab "
            "USING fts5vocab(main, products, row)"
        )
        self._vocab_table_ready = True

    def _ensure_ngram_index(self) -> None:
        if self._ngram_index is not None:
            return
        if self._provided_vocabulary is not None:
            raw_vocabulary = self._provided_vocabulary
        else:
            self._ensure_vocab_table()
            assert self.connection is not None
            raw_vocabulary = self.connection.execute(
                "SELECT term, doc FROM temp.product_vocab "
                "WHERE length(term) BETWEEN 4 AND 24 AND doc >= 2"
            ).fetchall()

        self._vocabulary = [
            (str(token), int(frequency))
            for token, frequency in raw_vocabulary
            if ASCII_WORD_RE.fullmatch(str(token))
        ]
        self._vocabulary_set = {token for token, _ in self._vocabulary}
        postings: defaultdict[str, list[int]] = defaultdict(list)
        for index, (token, _) in enumerate(self._vocabulary):
            for gram in character_ngrams(token):
                postings[gram].append(index)
        self._ngram_index = {
            gram: tuple(indexes) for gram, indexes in postings.items()
        }
