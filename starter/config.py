from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecommendationPolicy:
    """Limit early recommendation breadth while confidence is still evolving."""

    turn_limits: tuple[int, ...] = (1, 1, 3)

    def __post_init__(self) -> None:
        if any(limit < 1 for limit in self.turn_limits):
            raise ValueError("recommendation limits must be positive")

    def limit_for(self, turn: int, requested: int) -> int:
        requested = max(1, min(int(requested), 10))
        if 1 <= turn <= len(self.turn_limits):
            return min(requested, self.turn_limits[turn - 1])
        return requested


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Runtime feature policy; the evaluated default is deterministic and offline."""

    enable_vector_reranker: bool = False
    recommendation_policy: RecommendationPolicy = RecommendationPolicy()


DEFAULT_AGENT_CONFIG = AgentConfig()
FULL_BREADTH_POLICY = RecommendationPolicy(turn_limits=())
