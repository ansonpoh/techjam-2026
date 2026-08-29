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
EXPLORING_RE = re.compile(r"\bi(?:'m| am) still exploring\b", re.I)
QUESTION_BOILERPLATE_RE = re.compile(
    r"\b(?:options are not quite right|ask me about|closest matches (?:differ|vary)|"
    r"which .+ best fits what you need)\b",
    re.I,
)
GENERIC_PREFIX_RE = re.compile(
    r"^(?:(?:for that|actually),?\s+)?(?:i\s+)?(?:would\s+)?"
    r"(?:prefer|need|want|am looking for|(?:'m|am) looking for)\s+",
    re.I,
)
GENERIC_CATEGORY_ONLY = {"shoe", "shoes", "jewelry", "jewellery"}


@dataclass(frozen=True)
class Evidence:
    text: str
    weight: float
    source: str
    turn: int
    attribute: str | None = None


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

        is_override = bool(OVERRIDE_RE.search(message))
        if is_override:
            # The opening preference is superseded. Explicit clarification
            # answers remain valid unless the customer replaces them by name.
            self.evidence = [item for item in self.evidence if item.source != "initial_preference"]

        category_match = LOOKING_FOR_RE.search(message)
        if category_match and (not self.category_text or is_override):
            next_category = re.sub(r"\s+instead$", "", category_match.group(1), flags=re.I)
            self.category_text = _clean(next_category)
            if self.category_text:
                self.evidence = [item for item in self.evidence if item.source != "category"]
                self._add(self.category_text, 1.4, "category", turn)

        match = NEED_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 6.0, "override", turn)  # Boosted weight for faster recovery
            return

        match = REQUIREMENT_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 3.8, "hard_constraint", turn, self._answer_attribute())
            return

        match = MATTERS_RE.search(message)
        if match:
            for value in _split_constraints(match.group(1)):
                self._add(value, 3.3, "clarification", turn, self._answer_attribute())
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
            self._add(
                message,
                2.5 if turn > 1 else 2.0,
                "clarification",
                turn,
                self._answer_attribute(),
            )

    def _answer_attribute(self) -> str | None:
        return self.asked_attributes[-1] if self.asked_attributes else None

    def _add(
        self,
        text: str,
        weight: float,
        source: str,
        turn: int,
        attribute: str | None = None,
    ) -> None:
        text = _clean(text)
        if not text:
            return
        key = text.casefold()
        if any(item.text.casefold() == key and item.source == source for item in self.evidence):
            return
        self.evidence.append(
            Evidence(text=text, weight=weight, source=source, turn=turn, attribute=attribute)
        )

    def record_question(self, attribute: str) -> None:
        self.asked_attributes.append(attribute)

    @property
    def latest_evidence(self) -> Evidence | None:
        return self.evidence[-1] if self.evidence else None

    def semantic_query(self) -> str | None:
        """Return one concise intent query, or None when intent is only generic."""
        category = _clean(self.category_text)
        required: list[str] = []
        intended_use: list[str] = []
        seen: set[str] = {category.casefold()} if category else set()

        for item in self.evidence:
            if item.source == "category":
                if not category:
                    category = _clean(item.text)
                    if category:
                        seen.add(category.casefold())
                continue
            value = _clean(GENERIC_PREFIX_RE.sub("", item.text))
            if (
                not value
                or NO_PREFERENCE_RE.search(value)
                or EXPLORING_RE.search(value)
                or QUESTION_BOILERPLATE_RE.search(value)
                or (not category and value.casefold() in GENERIC_CATEGORY_ONLY)
                or value.casefold() in seen
            ):
                continue
            seen.add(value.casefold())
            target = intended_use if item.attribute == "use_case" else required
            target.append(value)

        # A category alone describes a catalog aisle, not a semantic target.
        if not required and not intended_use:
            return None

        lines: list[str] = []
        if category:
            lines.append(f"Product category: {category}")
        if required:
            lines.append(f"Required features: {'; '.join(required)}")
        if intended_use:
            lines.append(f"Intended use: {'; '.join(intended_use)}")
        return "\n".join(lines) or None
