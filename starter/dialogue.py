from __future__ import annotations

import re
from dataclasses import dataclass, field


OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|no longer|rather than)\b", re.I
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference\b",
    re.I,
)
LOOKING_FOR_RE = re.compile(r"\blooking for\s+(.+?)(?:[,.]|$)", re.I)
NEED_RE = re.compile(r"\bwhat i need is\s*:\s*(.+)$", re.I)
REQUIREMENT_RE = re.compile(r"\bkey requirement is\s*:\s*(.+)$", re.I)
MATTERS_RE = re.compile(r"\bwhat matters is\s*:\s*(.+)$", re.I)


@dataclass(frozen=True)
class Evidence:
    text: str
    weight: float
    source: str
    turn: int


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.")


def _split_constraints(value: str) -> list[str]:
    return [cleaned for part in value.split(";") if (cleaned := _clean(part))]


@dataclass
class SessionState:
    user_profile: dict
    evidence: list[Evidence] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    category_text: str = ""
    last_turn: int = 0

    def observe(self, message: str, turn: int) -> None:
        """Convert the latest customer message into weighted positive evidence."""
        if turn <= self.last_turn:
            return
        self.last_turn = turn
        message = _clean(str(message))
        self.messages.append(message)

        if NO_PREFERENCE_RE.search(message):
            if self.asked_attributes:
                self.no_preference_attributes.add(self.asked_attributes[-1])
            return

        if OVERRIDE_RE.search(message):
            # The opening preference is superseded. Explicit clarification
            # answers remain valid unless the customer replaces them by name.
            self.evidence = [item for item in self.evidence if item.source != "initial_preference"]

        category_match = LOOKING_FOR_RE.search(message)
        if category_match and not self.category_text:
            self.category_text = _clean(category_match.group(1))
            if self.category_text:
                self._add(self.category_text, 1.4, "category", turn)

        match = NEED_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 4.5, "override", turn)
            return

        match = REQUIREMENT_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 3.8, "hard_constraint", turn)
            return

        match = MATTERS_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 3.3, "clarification", turn)
            return

        if category_match:
            remainder = message[category_match.end():]
            remainder = re.sub(
                r"^(?:\s*but\s+)?i(?:'m| am) still exploring$", "", remainder, flags=re.I
            )
            remainder = _clean(remainder)
            if remainder:
                self._add(remainder, 1.8, "initial_preference", turn)
            return

        if not re.search(r"options are not quite right|ask me about", message, re.I):
            self._add(message, 2.5 if turn > 1 else 2.0, "clarification", turn)

    def _add(self, text: str, weight: float, source: str, turn: int) -> None:
        text = _clean(text)
        if not text:
            return
        key = text.casefold()
        if any(item.text.casefold() == key and item.source == source for item in self.evidence):
            return
        self.evidence.append(Evidence(text=text, weight=weight, source=source, turn=turn))

    def record_question(self, attribute: str) -> None:
        self.asked_attributes.append(attribute)

    @property
    def latest_evidence(self) -> Evidence | None:
        return self.evidence[-1] if self.evidence else None
