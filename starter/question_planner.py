from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from starter.dialogue import SessionState
from starter.product_features import FACET_ORDER, ProductFeatures, ProductFeatureStore


# Additional Information for agent to filter by.
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


class AdaptiveQuestionPlanner:
    """Select clarification facets from candidate-pool information gain."""

    def __init__(self, feature_store: ProductFeatureStore) -> None:
        self.feature_store = feature_store

    def choose(
        self,
        state: SessionState,
        candidates: list[dict],
        turn: int,
    ) -> tuple[str | None, str]:
        if turn >= 10 or not candidates:
            return None, "These are my best matches based on everything you've shared."

        # Cap to avoid repetition and honor an explicit no-preference reply.
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
            attribute: []
            for attribute in (*FACET_ORDER, "budget", "brand", "category", "feature")
        }

        documents = [self._features(product) for product in candidates]
        feature_frequency = Counter(
            token
            for document in documents
            for token in document.feature_tokens
        )

        prices = sorted(
            document.price
            for document in documents
            if document.price is not None
        )
        price_cuts = self._quartiles(prices)

        for product, document in zip(candidates, documents):
            question_features = self.feature_store.question_features(product)
            for attribute in FACET_ORDER:
                observations[attribute].append(
                    question_features.facet_values(attribute)
                )

            observations["budget"].append(
                self._budget_bucket(document.price, price_cuts)
            )
            observations["brand"].append(
                (document.brand,) if document.brand else ()
            )

            category = " ".join(
                document.category_tokens[-3:]
            )
            observations["category"].append((category,) if category else ())

            feature_values = sorted(
                document.feature_tokens,
                key=lambda token: (feature_frequency[token], token),
            )[:2]
            observations["feature"].append(
                tuple(feature_values)
            )

        return [
            self._information_gain(attribute, values)
            for attribute, values in observations.items()
        ]

    @staticmethod
    def _features(product: dict) -> ProductFeatures:
        features = product.get("_features")
        if not isinstance(features, ProductFeatures):
            raise TypeError("candidate is missing precomputed ProductFeatures")
        return features

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
        if cuts is None or value is None:
            return ()
        bucket = sum(value > cut for cut in cuts) + 1
        return (f"price group {bucket}",)
