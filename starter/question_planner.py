from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from starter.dialogue import SessionState
from starter.retrieval import STOPWORDS, TOKEN_RE


# The attribute names are imposed by the Agent API contract. The planner does
# not use a fixed ordering or fixed question sentences; it chooses among these
# facets from the live candidate distribution.
FACET_PATTERNS = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|"
        r"denim|fleece|suede|canvas|rubber|synthetic|acrylic|fabric)\b", re.I
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|"
        r"orange|beige|navy|gold|silver|multicolor)\b", re.I
    ),
    "size": re.compile(
        r"\b(x{0,3}s|x{0,4}l|small|medium|large|wide|narrow|petite|plus size)\b", re.I
    ),
    "style": re.compile(
        r"\b(casual|formal|classic|modern|vintage|slim|regular|relaxed|fitted|"
        r"loose|athletic|crew neck|v-neck|long sleeve|short sleeve)\b", re.I
    ),
    "use_case": re.compile(
        r"\b(running|hiking|walking|work|office|gym|workout|sports|travel|"
        r"winter|outdoor|wedding|party|sleep|swimming|cycling)\b", re.I
    ),
}

# Information gain is only useful when the customer is likely to have the
# requested detail. Product features, material, color, and price are common in
# shopping intent; catalog taxonomy and merchant names are much less likely to
# be meaningful customer preferences.
ANSWERABILITY_PRIORS = {
    "feature": 1.00,
    "material": 0.95,
    "color": 0.90,
    "budget": 0.80,
    "size": 0.70,
    "style": 0.65,
    "use_case": 0.60,
    "category": 0.30,
    "brand": 0.20,
}
EARLY_OPEN_QUESTION_LIMIT = 2


@dataclass(frozen=True)
class FacetScore:
    attribute: str
    information_gain: float
    examples: tuple[str, ...]


def _tokens(value: object) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(str(value or ""))
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def _compact(value: object, limit: int = 32) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit].rstrip()


class AdaptiveQuestionPlanner:
    """Select clarification facets from candidate-pool information gain."""

    def choose(
        self,
        state: SessionState,
        candidates: list[dict],
        turn: int,
    ) -> tuple[str | None, str]:
        if turn >= 10 or not candidates:
            return None, "These are my best matches based on everything you've shared."

        # Early in an uncertain search, a broad must-have question has the
        # highest chance of eliciting one of the customer's actual constraints.
        # Cap it to avoid repetition and honor an explicit no-preference reply.
        if (
            turn <= 3
            and "other" not in state.no_preference_attributes
            and state.asked_attributes.count("other") < EARLY_OPEN_QUESTION_LIMIT
        ):
            state.record_question("other")
            return "other", self._word_question("other", ())

        facet_scores = self._score_facets(candidates)
        available = [
            facet
            for facet in facet_scores
            if facet.attribute not in state.no_preference_attributes
        ]
        if not available:
            return None, "These are my best matches based on everything you've shared."

        adjusted = [
            FacetScore(
                facet.attribute,
                facet.information_gain
                * ANSWERABILITY_PRIORS.get(facet.attribute, 0.50)
                / (1.0 + 0.85 * state.asked_attributes.count(facet.attribute)),
                facet.examples,
            )
            for facet in available
        ]
        adjusted.sort(key=lambda facet: (-facet.information_gain, facet.attribute))

        attribute = adjusted[0].attribute
        top_facets = adjusted[:3]
        if self._needs_open_question(candidates, top_facets, state):
            attribute = "other"
            examples = tuple(facet.attribute.replace("_", " ") for facet in top_facets[:2])
        else:
            examples = adjusted[0].examples

        state.record_question(attribute)
        return attribute, self._word_question(attribute, examples)

    def _score_facets(self, candidates: list[dict]) -> list[FacetScore]:
        observations: dict[str, list[tuple[str, ...]]] = {
            attribute: [] for attribute in (*FACET_PATTERNS, "budget", "brand", "category", "feature")
        }

        feature_documents = [set(_tokens(product.get("features"))) for product in candidates]
        feature_frequency = Counter(token for document in feature_documents for token in document)

        prices = sorted(
            float(product["price"])
            for product in candidates
            if self._is_positive_number(product.get("price"))
        )
        price_cuts = self._quartiles(prices)

        for index, product in enumerate(candidates):
            searchable = " ".join(
                str(product.get(field) or "")
                for field in ("title", "features", "details", "description")
            )
            for attribute, pattern in FACET_PATTERNS.items():
                values = tuple(sorted({match.lower() for match in pattern.findall(searchable)}))
                observations[attribute].append(values)

            observations["budget"].append(self._budget_bucket(product.get("price"), price_cuts))
            brand = _compact(product.get("store")).casefold()
            observations["brand"].append((brand,) if brand else ())

            category_tokens = _tokens(product.get("categories"))
            category = " ".join(category_tokens[-3:])
            observations["category"].append((category,) if category else ())

            feature_values = sorted(
                feature_documents[index],
                key=lambda token: (feature_frequency[token], token),
            )[:2]
            observations["feature"].append(tuple(feature_values))

        return [
            self._information_gain(attribute, values)
            for attribute, values in observations.items()
        ]

    @staticmethod
    def _information_gain(
        attribute: str, observations: list[tuple[str, ...]]
    ) -> FacetScore:
        if not observations:
            return FacetScore(attribute, 0.0, ())
        signatures = [" / ".join(values) if values else "<unknown>" for values in observations]
        counts = Counter(signatures)
        total = len(signatures)
        gini_reduction = 1.0 - sum((count / total) ** 2 for count in counts.values())
        coverage = 1.0 - counts.get("<unknown>", 0) / total
        information_gain = coverage * gini_reduction
        examples = tuple(
            value
            for value, _ in counts.most_common()
            if value != "<unknown>"
        )[:3]
        return FacetScore(attribute, information_gain, examples)

    @staticmethod
    def _needs_open_question(
        candidates: list[dict], facets: list[FacetScore], state: SessionState
    ) -> bool:
        if len(facets) < 2:
            return False
        scores = [float(product.get("_rank_score") or 0.0) for product in candidates[:10]]
        relevance_spread = (
            (scores[0] - scores[-1]) / max(abs(scores[0]), 1.0)
            if len(scores) >= 2
            else 1.0
        )
        facet_competition = facets[1].information_gain / max(facets[0].information_gain, 1e-9)
        broad_uncertainty = relevance_spread < 0.20 and facet_competition > 0.72
        repeated_penalty = 1.0 + 0.70 * state.asked_attributes.count("other")
        return broad_uncertainty and facet_competition / repeated_penalty > 0.40

    @staticmethod
    def _word_question(attribute: str, examples: tuple[str, ...]) -> str:
        if attribute == "other":
            dimensions = " and ".join(examples) if examples else "several details"
            return (
                f"The closest matches vary across {dimensions}. "
                "What must-have detail should I prioritize to narrow them down?"
            )
        label = attribute.replace("_", " ")
        usable_examples = [value for value in examples if len(value) <= 28][:3]
        example_text = f"—for example, {', '.join(usable_examples)}" if usable_examples else ""
        return (
            f"The closest matches differ by {label}{example_text}. "
            f"Which {label} best fits what you need?"
        )

    @staticmethod
    def _is_positive_number(value: object) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float, float] | None:
        if len(values) < 4:
            return None
        return (
            values[len(values) // 4],
            values[len(values) // 2],
            values[(3 * len(values)) // 4],
        )

    @staticmethod
    def _budget_bucket(
        value: object, cuts: tuple[float, float, float] | None
    ) -> tuple[str, ...]:
        if cuts is None or not AdaptiveQuestionPlanner._is_positive_number(value):
            return ()
        price = float(value)
        bucket = sum(price > cut for cut in cuts) + 1
        return (f"price group {bucket}",)
