"""Tests for LLM-based slot extraction."""

import unittest
from unittest.mock import Mock, patch
import json

from starter.llm_extractor import (
    LLMSlotExtractor,
    LLMExtractionResult,
    StateMachineWithLLM,
    EXTRACTION_PROMPT,
)


class MockLLMExtractor(LLMSlotExtractor):
    """Mock extractor that returns predefined responses."""

    def __init__(self, responses: dict[str, dict] = None):
        """
        Initialize with predefined responses.

        Args:
            responses: dict mapping message patterns to response dicts
        """
        self.provider = "mock"
        self.model = "mock-model"
        self.responses = responses or {}
        self.call_count = 0

    def extract(self, message: str, current_slots: dict = None) -> LLMExtractionResult:
        """Return predefined response based on message."""
        self.call_count += 1

        # Find matching response
        for pattern, response in self.responses.items():
            if pattern.lower() in message.lower():
                return LLMExtractionResult(
                    slots=response.get("slots", {}),
                    override_type=response.get("override_type"),
                    cleared_attributes=response.get("cleared_attributes", []),
                    confidence=response.get("confidence", 0.9),
                    usage={"prompt_tokens": 50, "completion_tokens": 20},
                    raw_response=json.dumps(response)
                )

        # Default empty response
        return LLMExtractionResult(
            usage={"prompt_tokens": 50, "completion_tokens": 10}
        )


class LLMExtractionResultTest(unittest.TestCase):
    """Test LLMExtractionResult dataclass."""

    def test_default_values(self):
        """Default values are properly initialized."""
        result = LLMExtractionResult()
        self.assertEqual(result.slots, {})
        self.assertIsNone(result.override_type)
        self.assertEqual(result.cleared_attributes, [])
        self.assertEqual(result.confidence, 0.0)

    def test_with_values(self):
        """Values are properly stored."""
        result = LLMExtractionResult(
            slots={"color": "blue"},
            override_type="partial",
            confidence=0.95
        )
        self.assertEqual(result.slots, {"color": "blue"})
        self.assertEqual(result.override_type, "partial")
        self.assertEqual(result.confidence, 0.95)


class StateMachineWithLLMTest(unittest.TestCase):
    """Test StateMachineWithLLM state management."""

    def setUp(self):
        """Set up mock extractor with predefined responses."""
        self.responses = {
            "blue cotton": {
                "slots": {"color": "blue", "material": "cotton"},
                "confidence": 0.9
            },
            "blue shoes": {
                "slots": {"color": "blue"},
                "confidence": 0.9
            },
            "for hiking": {
                "slots": {"use_case": "hiking"},
                "confidence": 0.85
            },
            "under $50": {
                "slots": {"budget": "under $50"},
                "confidence": 0.9
            },
            "instead of blue": {
                "slots": {"color": "navy"},
                "override_type": "partial",
                "cleared_attributes": ["color"],
                "confidence": 0.95
            },
            "changed my mind": {
                "slots": {"material": "leather"},
                "override_type": "full",
                "confidence": 0.9
            },
        }
        self.extractor = MockLLMExtractor(self.responses)
        self.state = StateMachineWithLLM(self.extractor)
        self.state.reset("test_session", {})

    def test_incremental_slot_filling(self):
        """Slots accumulate across turns."""
        # Turn 1: Extract color and material
        result = self.state.observe("I want blue cotton shoes", turn=1)
        self.assertIn("color", result["slots_updated"])
        self.assertIn("material", result["slots_updated"])

        slots = self.state.get_active_constraints()
        self.assertEqual(slots["color"], "blue")
        self.assertEqual(slots["material"], "cotton")

        # Turn 2: Add use case
        result = self.state.observe("for hiking", turn=2)
        self.assertIn("use_case", result["slots_updated"])

        slots = self.state.get_active_constraints()
        self.assertEqual(slots["color"], "blue")  # Preserved
        self.assertEqual(slots["material"], "cotton")  # Preserved
        self.assertEqual(slots["use_case"], "hiking")  # Added

        # Turn 3: Add budget
        result = self.state.observe("under $50", turn=3)

        slots = self.state.get_active_constraints()
        self.assertEqual(len(slots), 4)
        self.assertEqual(slots["budget"], "under $50")

    def test_partial_override(self):
        """Partial override clears specific slot and adds new value."""
        # Setup initial state
        self.state.observe("blue cotton shoes", turn=1)
        self.assertEqual(self.state.slots["color"]["value"], "blue")

        # Partial override
        result = self.state.observe("instead of blue, make it navy", turn=2)

        self.assertEqual(result["override_type"], "partial")
        self.assertIn("color", result["slots_cleared"])
        self.assertIn("color", result["slots_updated"])

        slots = self.state.get_active_constraints()
        self.assertEqual(slots["color"], "navy")
        self.assertEqual(slots["material"], "cotton")  # Preserved

    def test_full_override(self):
        """Full override clears early slots."""
        # Setup initial state
        self.state.observe("blue cotton shoes", turn=1)
        self.state.observe("for hiking", turn=2)

        initial_slots = self.state.get_active_constraints()
        self.assertEqual(len(initial_slots), 3)

        # Full override
        result = self.state.observe("changed my mind, I want leather", turn=3)

        self.assertEqual(result["override_type"], "full")
        self.assertIn("color", result["slots_cleared"])
        self.assertIn("material", result["slots_cleared"])

        slots = self.state.get_active_constraints()
        self.assertEqual(slots["material"], "leather")
        self.assertNotIn("color", slots)

    def test_slot_history_tracking(self):
        """Slot changes are recorded in history."""
        self.state.observe("blue shoes", turn=1)
        self.state.observe("instead of blue, navy", turn=2)

        # Check history
        history = self.state.slot_history
        self.assertTrue(len(history) >= 2)

        # Find the clear and set for color
        color_events = [h for h in history if h["attribute"] == "color"]
        actions = [h["action"] for h in color_events]
        self.assertIn("clear", actions)
        self.assertIn("set", actions)

    def test_token_usage_tracking(self):
        """Token usage is accumulated."""
        self.state.observe("blue shoes", turn=1)
        self.state.observe("for hiking", turn=2)

        self.assertGreater(self.state.total_usage["prompt_tokens"], 0)
        self.assertGreater(self.state.total_usage["completion_tokens"], 0)

    def test_duplicate_turn_ignored(self):
        """Same turn number should not process twice."""
        self.state.observe("blue shoes", turn=1)
        initial_count = self.extractor.call_count

        self.state.observe("red shoes", turn=1)  # Same turn

        # Should not have made another LLM call
        self.assertEqual(self.extractor.call_count, initial_count)

    def test_get_search_terms(self):
        """Search terms are weighted by confidence."""
        self.state.observe("blue cotton", turn=1)

        terms = self.state.get_search_terms()
        self.assertEqual(len(terms), 2)

        # Each term should be (value, confidence)
        values = [t[0] for t in terms]
        self.assertIn("blue", values)
        self.assertIn("cotton", values)


class PromptTemplateTest(unittest.TestCase):
    """Test the extraction prompt template."""

    def test_prompt_includes_attributes(self):
        """Prompt includes valid attributes."""
        self.assertIn("material", EXTRACTION_PROMPT)
        self.assertIn("color", EXTRACTION_PROMPT)
        self.assertIn("budget", EXTRACTION_PROMPT)

    def test_prompt_has_examples(self):
        """Prompt includes examples."""
        self.assertIn("instead of", EXTRACTION_PROMPT.lower())
        self.assertIn("changed my mind", EXTRACTION_PROMPT.lower())

    def test_prompt_formatting(self):
        """Prompt can be formatted with parameters."""
        formatted = EXTRACTION_PROMPT.format(
            current_slots='{"color": "blue"}',
            message="I want red shoes",
            attributes="color, material, size"
        )
        self.assertIn("blue", formatted)
        self.assertIn("red shoes", formatted)


class IntegrationScenarioTest(unittest.TestCase):
    """Integration tests simulating real conversation flows."""

    def setUp(self):
        self.responses = {
            # Scenario 1: Buying flow
            "red leather jacket": {
                "slots": {"color": "red", "material": "leather"},
                "confidence": 0.95
            },
            "for work": {
                "slots": {"use_case": "work"},
                "confidence": 0.9
            },
            # Scenario 2: Browsing with refinement
            "looking for dress": {
                "slots": {"category": "dress"},
                "confidence": 0.7
            },
            "prefer blue": {
                "slots": {"color": "blue"},
                "confidence": 0.85
            },
            "something casual": {
                "slots": {"style": "casual"},
                "confidence": 0.8
            },
            # Override scenarios
            "actually green": {
                "slots": {"color": "green"},
                "override_type": "partial",
                "cleared_attributes": ["color"],
                "confidence": 0.9
            },
        }
        self.extractor = MockLLMExtractor(self.responses)
        self.state = StateMachineWithLLM(self.extractor)

    def test_buying_scenario_accumulation(self):
        """Buying scenario: constraints accumulate quickly."""
        self.state.reset("buying_001", {"preference_tags": []})

        # Turn 1: Clear initial constraint
        self.state.observe("I need a red leather jacket", turn=1)
        slots = self.state.get_active_constraints()
        self.assertEqual(slots.get("color"), "red")
        self.assertEqual(slots.get("material"), "leather")

        # Turn 2: Add use case
        self.state.observe("for work", turn=2)
        slots = self.state.get_active_constraints()
        self.assertEqual(slots.get("use_case"), "work")
        self.assertEqual(len(slots), 3)

    def test_browsing_scenario_gradual(self):
        """Browsing scenario: constraints added gradually."""
        self.state.reset("browsing_001", {"preference_tags": []})

        # Turn 1: Vague
        self.state.observe("I'm looking for dress", turn=1)
        self.assertEqual(len(self.state.get_active_constraints()), 1)

        # Turn 2: Color
        self.state.observe("I prefer blue", turn=2)
        self.assertEqual(len(self.state.get_active_constraints()), 2)

        # Turn 3: Style
        self.state.observe("something casual", turn=3)
        slots = self.state.get_active_constraints()
        self.assertEqual(len(slots), 3)
        self.assertEqual(slots["style"], "casual")

    def test_intent_override_scenario(self):
        """Intent override: customer changes preference."""
        self.state.reset("override_001", {"preference_tags": []})

        # Initial preference
        self.state.observe("I prefer blue dress", turn=1)
        self.assertEqual(self.state.get_active_constraints().get("color"), "blue")

        # Override on turn 3
        self.state.observe("actually green would be better", turn=3)
        slots = self.state.get_active_constraints()
        self.assertEqual(slots.get("color"), "green")


if __name__ == "__main__":
    unittest.main()
