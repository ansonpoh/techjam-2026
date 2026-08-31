from pathlib import Path

from starter.config import AgentConfig, DEFAULT_AGENT_CONFIG
from starter.dialogue import SessionState
from starter.memory import UserProfileStore
from starter.question_planner import AdaptiveQuestionPlanner
from starter.ranking import DEFAULT_RANKING_POLICIES, RankingPolicies
from starter.retrieval import FEATURE_CACHE_SIZE, CatalogSearch
from starter.vector_index import VectorIndex


AMBIGUOUS_FIELD_SCORE_THRESHOLD = 2.0


class Agent:
    """Deterministic conversational product-search agent."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        feature_cache_size: int = FEATURE_CACHE_SIZE,
        *,
        config: AgentConfig = DEFAULT_AGENT_CONFIG,
        ranking_policies: RankingPolicies = DEFAULT_RANKING_POLICIES,
        vector_index: VectorIndex | None = None,
        catalog_index_path: str | Path | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config
        self.search = CatalogSearch(
            self.catalog_path,
            feature_cache_size=feature_cache_size,
            enable_vector_reranker=config.enable_vector_reranker,
            ranking_policies=ranking_policies,
            vector_index=vector_index,
            catalog_index_path=catalog_index_path,
        )
        self.question_planner = AdaptiveQuestionPlanner(self.search.feature_store)
        self._sessions: dict[str, SessionState] = {}
        self._ambiguity_deferred: set[str] = set()
        self.profile_store = UserProfileStore()

    def close(self) -> None:
        self.search.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile_id = str(user_profile.get("profile_id") or user_profile.get("user_id") or session_id)
        profile = self.profile_store.get(profile_id, user_profile)
        self._sessions[session_id] = SessionState(user_profile=user_profile, long_term_profile=profile)
        self._ambiguity_deferred.discard(session_id)

    def export_profile(self, profile_id: str) -> dict | None:
        profile = self.profile_store.profiles.get(profile_id)
        return profile.snapshot() if profile else None

    def forget_profile(self, profile_id: str) -> None:
        self.profile_store.forget(profile_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        state.observe(user_message, turn)
        result = self.search.search_with_context(
            state, limit=max(1, min(int(top_k), 10))
        )

        question_plan = self.question_planner.choose(
            state, result.candidates, turn
        )

        top_candidate = result.candidates[0] if result.candidates else {}
        hard_count = int(top_candidate.get("_hard_constraint_count") or 0)
        exact_count = int(top_candidate.get("_hard_constraint_exact_count") or 0)
        clarification_count = (
            state.asked_attributes.count(question_plan.attribute)
            if question_plan.attribute
            else 0
        )
        recommendation_limit = self.config.recommendation_policy.limit_for(
            turn,
            top_k,
            scores=tuple(score for _, score in result.recommendations),
            hard_constraint_coverage=(exact_count / hard_count if hard_count else 0.0),
            has_hard_constraints=hard_count > 0,
            has_answerable_clarification=(
                question_plan.attribute is not None
                and clarification_count <= 2
                and len(state.asked_attributes) <= 2
            ),
            clarification_expected_value=question_plan.expected_value,
            turns_remaining=max(0, 10 - turn),
        )
        # When the leading catalog records are observational siblings, their
        # popularity score is not evidence that one satisfies the request
        # better. Use the already-planned clarification before exposing an
        # arbitrary sibling order; the next answer commonly supplies the rare
        # feature phrase that disambiguates the records.
        unresolved_siblings = (
            recommendation_limit > 1
            and session_id not in self._ambiguity_deferred
            and turn < 9
            and question_plan.attribute is not None
            and len(result.candidates) >= 2
            and float(
                (result.candidates[0].get("_catalog_tiebreak") or (0.0,))[0]
            ) < AMBIGUOUS_FIELD_SCORE_THRESHOLD
            and result.candidates[0].get("_catalog_tiebreak")
            == result.candidates[1].get("_catalog_tiebreak")
            and result.candidates[0].get("_hard_constraint_exact_count")
            == result.candidates[1].get("_hard_constraint_exact_count")
            and result.candidates[0].get("_category_leaf_match")
            == result.candidates[1].get("_category_leaf_match")
        )
        if unresolved_siblings:
            self._ambiguity_deferred.add(session_id)
        ranked = (
            []
            if unresolved_siblings
            else result.recommendations[:recommendation_limit]
        )
        return {
            "message": question_plan.message,
            "ask_attribute": question_plan.attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 6)}
                for parent_asin, score in ranked
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": 0,
            },
        }
