"""LLM-based slot extraction for dynamic state updates.

Uses an LLM to parse user messages and extract structured slots,
handling nuanced language, negations, and intent overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

# Load .env file if present
def _load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

# Valid attributes per API contract
VALID_ATTRIBUTES = [
    "material", "color", "size", "style", "use_case",
    "budget", "brand", "category", "feature"
]

EXTRACTION_PROMPT = """You are a slot extraction system for a shopping assistant. Extract structured constraints from the user's message.

Current conversation state (slots already filled):
{current_slots}

User message: "{message}"

Extract any NEW or CHANGED constraints. Return a JSON object with:
{{
  "slots": {{
    "<attribute>": "<value>",
    ...
  }},
  "override": {{
    "type": "full" | "partial" | null,
    "cleared_attributes": ["<attr1>", ...]  // only for partial override
  }},
  "confidence": 0.0-1.0
}}

Valid attributes: {attributes}

Rules:
1. Only extract explicitly stated constraints
2. For "instead of X, Y" - set override.type="partial", cleared_attributes=[attr of X], and slots[attr]=Y
3. For "actually/changed my mind" - set override.type="full" (clears early slots)
4. Handle negations: "not blue" means DO NOT extract blue
5. Budget: extract as "under $X", "around $X", or "$X"
6. If no extractable constraints, return empty slots {{}}

Examples:
- "I want blue cotton shoes" → {{"slots": {{"color": "blue", "material": "cotton"}}, "override": {{"type": null}}, "confidence": 0.9}}
- "Instead of black, make it navy" → {{"slots": {{"color": "navy"}}, "override": {{"type": "partial", "cleared_attributes": ["color"]}}, "confidence": 0.95}}
- "Actually, I changed my mind. I want leather boots" → {{"slots": {{"material": "leather"}}, "override": {{"type": "full"}}, "confidence": 0.9}}
- "Not too expensive, maybe under $50" → {{"slots": {{"budget": "under $50"}}, "override": {{"type": null}}, "confidence": 0.85}}

Return ONLY the JSON object, no other text."""


@dataclass
class LLMExtractionResult:
    """Result from LLM slot extraction."""
    slots: dict[str, str] = field(default_factory=dict)
    override_type: Optional[str] = None  # "full", "partial", or None
    cleared_attributes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: str = ""


class LLMSlotExtractor:
    """
    LLM-based slot extractor for dynamic constraint parsing.

    Supports multiple LLM providers via environment variables:
    - OPENAI_API_KEY: Use OpenAI (gpt-4o-mini)
    - ANTHROPIC_API_KEY: Use Anthropic (claude-3-haiku)
    """

    def __init__(self, provider: str = "auto", model: str = None):
        """
        Initialize LLM extractor.

        Args:
            provider: "openai", "anthropic", or "auto" (detect from env)
            model: Override default model selection
        """
        self.provider = self._detect_provider(provider)
        self.model = model or self._default_model()
        self._client = None

    def _detect_provider(self, provider: str) -> str:
        """Detect available LLM provider from environment."""
        if provider != "auto":
            return provider

        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        # Check both common OpenAI key names
        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        return "none"

    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "anthropic":
            return "claude-3-haiku-20240307"
        if self.provider == "openai":
            return "gpt-4o-mini"
        return ""

    def _get_client(self):
        """Lazy-load the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        elif self.provider == "openai":
            try:
                import openai
                # Support both OPENAI_API_KEY and OPENAI_APIKEY
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            raise ValueError(f"No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")

        return self._client

    def extract(
        self,
        message: str,
        current_slots: dict[str, str] = None
    ) -> LLMExtractionResult:
        """
        Extract slots from user message using LLM.

        Args:
            message: User's message
            current_slots: Current slot state for context

        Returns:
            LLMExtractionResult with extracted slots and metadata
        """
        if self.provider == "none":
            # Fallback to empty result if no LLM available
            return LLMExtractionResult()

        current_slots = current_slots or {}

        prompt = EXTRACTION_PROMPT.format(
            current_slots=json.dumps(current_slots) if current_slots else "{}",
            message=message,
            attributes=", ".join(VALID_ATTRIBUTES)
        )

        try:
            if self.provider == "anthropic":
                return self._extract_anthropic(prompt)
            elif self.provider == "openai":
                return self._extract_openai(prompt)
        except Exception as e:
            # Return empty result on error
            return LLMExtractionResult(raw_response=f"Error: {e}")

        return LLMExtractionResult()

    def _extract_anthropic(self, prompt: str) -> LLMExtractionResult:
        """Extract using Anthropic API."""
        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens
        }

        return self._parse_response(raw, usage)

    def _extract_openai(self, prompt: str) -> LLMExtractionResult:
        """Extract using OpenAI API."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }

        return self._parse_response(raw, usage)

    def _parse_response(self, raw: str, usage: dict) -> LLMExtractionResult:
        """Parse LLM response into structured result."""
        result = LLMExtractionResult(raw_response=raw, usage=usage)

        try:
            # Handle potential markdown code blocks
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            # Extract slots
            slots = data.get("slots", {})
            # Validate attributes
            result.slots = {
                k: v for k, v in slots.items()
                if k in VALID_ATTRIBUTES and isinstance(v, str) and v
            }

            # Extract override info
            override = data.get("override", {})
            if isinstance(override, dict):
                result.override_type = override.get("type")
                result.cleared_attributes = override.get("cleared_attributes", [])

            # Extract confidence
            result.confidence = float(data.get("confidence", 0.8))

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return result


class StateMachineWithLLM:
    """
    ConversationState manager that uses LLM for slot extraction.

    This wraps the LLM extractor and manages state updates.
    """

    def __init__(self, extractor: LLMSlotExtractor = None):
        """Initialize with optional LLM extractor."""
        self.extractor = extractor or LLMSlotExtractor()
        self.slots: dict[str, dict] = {}  # attr -> {value, confidence, turn, source}
        self.slot_history: list[dict] = []
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_turn: int = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Reset state for new session."""
        self.session_id = session_id
        self.user_profile = user_profile
        self.slots.clear()
        self.slot_history.clear()
        self.last_turn = 0

    def observe(self, message: str, turn: int) -> dict:
        """
        Process user message and update slots using LLM.

        Returns dict with:
        - slots_updated: list of updated slot names
        - slots_cleared: list of cleared slot names
        - override_type: "full", "partial", or None
        - usage: token usage for this call
        """
        if turn <= self.last_turn:
            return {"slots_updated": [], "slots_cleared": [], "override_type": None, "usage": {}}

        self.last_turn = turn
        current_slots = {k: v["value"] for k, v in self.slots.items()}

        # Extract using LLM
        result = self.extractor.extract(message, current_slots)

        # Track token usage
        self.total_usage["prompt_tokens"] += result.usage.get("prompt_tokens", 0)
        self.total_usage["completion_tokens"] += result.usage.get("completion_tokens", 0)

        slots_updated = []
        slots_cleared = []

        # Handle override
        if result.override_type == "full":
            # Clear all early slots
            for attr in list(self.slots.keys()):
                if self.slots[attr].get("turn", 0) <= 2:
                    self._record_clear(attr, turn)
                    slots_cleared.append(attr)
                    del self.slots[attr]
        elif result.override_type == "partial":
            # Clear specified attributes
            for attr in result.cleared_attributes:
                if attr in self.slots:
                    self._record_clear(attr, turn)
                    slots_cleared.append(attr)
                    del self.slots[attr]

        # Apply new slots
        for attr, value in result.slots.items():
            old_value = self.slots.get(attr, {}).get("value")
            self.slots[attr] = {
                "value": value,
                "confidence": result.confidence,
                "turn": turn,
                "source": "llm"
            }
            self._record_update(attr, old_value, value, turn)
            slots_updated.append(attr)

        return {
            "slots_updated": slots_updated,
            "slots_cleared": slots_cleared,
            "override_type": result.override_type,
            "usage": result.usage
        }

    def _record_clear(self, attr: str, turn: int) -> None:
        """Record slot clear in history."""
        old_value = self.slots.get(attr, {}).get("value")
        self.slot_history.append({
            "action": "clear",
            "attribute": attr,
            "old_value": old_value,
            "turn": turn
        })

    def _record_update(self, attr: str, old_value: Optional[str], new_value: str, turn: int) -> None:
        """Record slot update in history."""
        self.slot_history.append({
            "action": "update" if old_value else "set",
            "attribute": attr,
            "old_value": old_value,
            "new_value": new_value,
            "turn": turn
        })

    def get_active_constraints(self) -> dict[str, str]:
        """Get current slot values."""
        return {k: v["value"] for k, v in self.slots.items()}

    def get_search_terms(self) -> list[tuple[str, float]]:
        """Get weighted search terms from slots."""
        return [
            (slot["value"], slot["confidence"])
            for slot in self.slots.values()
        ]
