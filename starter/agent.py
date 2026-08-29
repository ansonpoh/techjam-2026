import re
from pathlib import Path
from typing import Optional

from starter.config import AgentConfig, DEFAULT_AGENT_CONFIG
from starter.dialogue import SessionState
from starter.question_planner import AdaptiveQuestionPlanner
from starter.ranking import DEFAULT_RANKING_POLICIES, RankingPolicies
from starter.retrieval import FEATURE_CACHE_SIZE, CatalogSearch
from starter.vector_index import VectorIndex
from starter.llm_extractor import LLMSlotExtractor, StateMachineWithLLM

# Regex fallback for override detection
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|no longer|rather than)\b", re.I
)


class Agent:
    """Conversational product-search agent with LLM-based slot extraction."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        feature_cache_size: int = FEATURE_CACHE_SIZE,
        *,
        config: AgentConfig = DEFAULT_AGENT_CONFIG,
        ranking_policies: RankingPolicies = DEFAULT_RANKING_POLICIES,
        vector_index: VectorIndex | None = None,
        llm_provider: str = "auto",
        llm_model: Optional[str] = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config
        self.search = CatalogSearch(
            self.catalog_path,
            feature_cache_size=feature_cache_size,
            enable_vector_reranker=config.enable_vector_reranker,
            ranking_policies=ranking_policies,
            vector_index=vector_index,
        )
        self.question_planner = AdaptiveQuestionPlanner(self.search.feature_store)

        # LLM-based slot extraction
        self._extractor = LLMSlotExtractor(provider=llm_provider, model=llm_model)

        # State tracking
        self._sessions: dict[str, SessionState] = {}  # Legacy (for search compatibility)
        self._llm_states: dict[str, StateMachineWithLLM] = {}  # LLM-based slots

    def close(self) -> None:
        self.search.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Initialize legacy state (still needed for search)
        self._sessions[session_id] = SessionState(user_profile=user_profile)

        # Initialize LLM-based state
        llm_state = StateMachineWithLLM(self._extractor)
        llm_state.reset(session_id, user_profile)
        self._llm_states[session_id] = llm_state

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        llm_state = self._llm_states.get(session_id)
        if state is None or llm_state is None:
            raise RuntimeError("reset must be called before respond")

        # Update legacy state (for search compatibility)
        state.observe(user_message, turn)

        # Update LLM-based state
        llm_result = llm_state.observe(user_message, turn)

        # Use legacy search as primary
        result = self.search.search_with_context(
            state, limit=max(1, min(int(top_k), 10))
        )

        # Detect override (LLM or regex fallback) and always rerank
        override_detected = (
            llm_result.get("override_type") or
            bool(OVERRIDE_RE.search(user_message))
        )
        if override_detected and llm_state.slots:
            result = self._rerank_with_llm_slots(result, llm_state)

        # Choose next question
        ask_attribute, message = self.question_planner.choose(
            state, result.candidates, turn
        )

        # Adapt message if override detected
        if override_detected:
            message = self._adapt_override_message(
                message,
                llm_result.get("slots_cleared", []),
                llm_result.get("slots_updated", []),
                llm_state
            )

        # Calculate token usage
        total_usage = llm_state.total_usage

        recommendation_limit = self.config.recommendation_policy.limit_for(
            turn, top_k
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
                "prompt_tokens": total_usage.get("prompt_tokens", 0) + result.prompt_tokens,
                "completion_tokens": total_usage.get("completion_tokens", 0),
            },
        }

    def _rerank_with_llm_slots(self, result, llm_state: StateMachineWithLLM):
        """Rerank results using LLM-extracted slots."""
        from starter.retrieval import SearchResult, terms, FIELD_WEIGHTS

        if not result.candidates:
            return result

        # Get slot values and weights
        slot_terms = llm_state.get_search_terms()
        if not slot_terms:
            return result

        # Rerank candidates based on slot matching
        reranked = []
        for product in result.candidates:
            # Build product text
            product_text = " ".join(
                str(product.get(field) or "").lower()
                for field in FIELD_WEIGHTS
            )
            product_tokens = set(terms(product_text))

            # Calculate slot match score
            slot_score = 0.0
            for value, confidence in slot_terms:
                value_tokens = set(terms(value))
                if value_tokens:
                    match_ratio = len(value_tokens & product_tokens) / len(value_tokens)
                    slot_score += confidence * match_ratio * 2.0

            # Combine with original score
            original_score = product.get("_rank_score", 0.0)
            combined_score = 0.6 * original_score + 0.4 * slot_score * 10

            reranked.append((product["parent_asin"], combined_score, product))

        reranked.sort(key=lambda x: (-x[1], x[0]))

        recommendations = [(asin, score) for asin, score, _ in reranked[:10]]
        candidates = [p for _, _, p in reranked[:100]]

        return SearchResult(recommendations=recommendations, candidates=candidates)

    def _adapt_override_message(
        self,
        base_message: str,
        cleared: list[str],
        updated: list[str],
        llm_state: StateMachineWithLLM
    ) -> str:
        """Acknowledge intent override in the response message."""
        if cleared and updated:
            new_values = ", ".join(
                llm_state.slots[attr]["value"]
                for attr in updated
                if attr in llm_state.slots
            )
            if new_values:
                return f"Got it, focusing on {new_values} now. {base_message}"
        elif updated:
            new_values = ", ".join(
                llm_state.slots[attr]["value"]
                for attr in updated
                if attr in llm_state.slots
            )
            if new_values:
                return f"Understood, looking for {new_values}. {base_message}"
        return base_message
