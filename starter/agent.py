from pathlib import Path

from starter.config import AgentConfig, DEFAULT_AGENT_CONFIG
from starter.dialogue import SessionState
from starter.memory import UserProfileStore
from starter.question_planner import AdaptiveQuestionPlanner
from starter.ranking import DEFAULT_RANKING_POLICIES, RankingPolicies
from starter.retrieval import FEATURE_CACHE_SIZE, CatalogSearch
from starter.vector_index import VectorIndex


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
        self.profile_store = UserProfileStore()

    def close(self) -> None:
        self.search.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile_id = str(user_profile.get("profile_id") or user_profile.get("user_id") or session_id)
        profile = self.profile_store.get(profile_id, user_profile)
        self._sessions[session_id] = SessionState(user_profile=user_profile, long_term_profile=profile)

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

        ask_attribute, message = self.question_planner.choose(
            state, result.candidates, turn
        )

        top_candidate = result.candidates[0] if result.candidates else {}
        hard_count = int(top_candidate.get("_hard_constraint_count") or 0)
        exact_count = int(top_candidate.get("_hard_constraint_exact_count") or 0)
        clarification_count = (
            state.asked_attributes.count(ask_attribute) if ask_attribute else 0
        )
        recommendation_limit = self.config.recommendation_policy.limit_for(
            turn,
            top_k,
            scores=tuple(score for _, score in result.recommendations),
            hard_constraint_coverage=(exact_count / hard_count if hard_count else 0.0),
            has_hard_constraints=hard_count > 0,
            has_answerable_clarification=(
                ask_attribute is not None
                and clarification_count <= 2
                and len(state.asked_attributes) <= 2
            ),
            turns_remaining=max(0, 10 - turn),
        )
        ranked = result.recommendations[:recommendation_limit]
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 6)}
                for parent_asin, score in ranked
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": 0,
            },
        }
