from pathlib import Path

from starter.dialogue import SessionState
from starter.question_planner import AdaptiveQuestionPlanner
from starter.retrieval import FEATURE_CACHE_SIZE, CatalogSearch


class Agent:
    """Stateful, offline conversational product-search agent."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        feature_cache_size: int = FEATURE_CACHE_SIZE,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.search = CatalogSearch(
            self.catalog_path, feature_cache_size=feature_cache_size
        )
        self.question_planner = AdaptiveQuestionPlanner(self.search.feature_store)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile=user_profile)

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
        
        ranked = result.recommendations[:1] if turn == 1 else result.recommendations
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 6)}
                for parent_asin, score in ranked
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
