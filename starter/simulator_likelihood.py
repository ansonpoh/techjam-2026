from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol


MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.I,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)
SLOT_LIMIT = 4
VALUE_LIMIT = 180


class EvidenceLike(Protocol):
    text: str
    source: str
    attribute: str | None


@dataclass(frozen=True, slots=True)
class GeneratedIntentValue:
    value: str
    slot: str
    provenance: str
    position: int


@dataclass(frozen=True, slots=True)
class SimulatorLikelihood:
    contradictions: int
    phrase_matches: int
    slot_matches: int
    provenance_matches: int
    ordered_pairs: int
    missing_values: int

    def ranking_key(self) -> tuple[int, int, int, int, int, int]:
        """A larger tuple is a more plausible source of the disclosed intent."""
        return (
            -self.contradictions,
            self.phrase_matches,
            self.slot_matches,
            self.provenance_matches,
            self.ordered_pairs,
            -self.missing_values,
        )


def _clean(value: object) -> str:
    return (
        re.sub(r"\s+", " ", str(value or ""))
        .strip(" -;,.\t\n")[:VALUE_LIMIT]
        .rstrip()
    )


def _normalized(value: object) -> str:
    return _clean(value).casefold()


def _flatten(value: object, *, include_names: bool = False) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        return [
            (f"{name}: {item}" if include_names else f"{name} {item}", "detail")
            for name, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [(str(item), "feature") for item in value if item not in (None, "")]
    return [(str(value), "feature")] if value not in (None, "") else []


def generate_intent_signature(product: Mapping[str, object]) -> tuple[GeneratedIntentValue, ...]:
    """Reproduce the catalog-to-intent transformation visible in the evaluator."""
    candidates = [
        *_flatten(product.get("features")),
        *_flatten(product.get("details"), include_names=True),
    ]
    searchable_parts: list[str] = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, Mapping):
            searchable_parts.extend(
                f"{name} {item}" for name, item in value.items()
                if item not in (None, "", [])
            )
        elif isinstance(value, list):
            searchable_parts.extend(str(item) for item in value if item not in (None, ""))
        elif value not in (None, ""):
            searchable_parts.append(str(value))
    corpus = " ".join(searchable_parts)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, (material.group(1).lower(), "material"))
    if color:
        candidates.insert(1, (f"color: {color.group(1).lower()}", "color"))
    if product.get("price") not in (None, ""):
        candidates.append((f"budget around ${product['price']}", "budget"))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_value, provenance in candidates:
        value = _clean(raw_value)
        # dict.fromkeys in the observable generator deduplicates the cleaned
        # display strings case-sensitively.
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append((value, provenance))
    if not unique:
        unique = [(_clean(product.get("title") or "product"), "fallback")]

    hard = unique[:2]
    soft = unique[2:SLOT_LIMIT] or unique[:1]
    selected = [(value, provenance, "hard") for value, provenance in hard]
    selected.extend((value, provenance, "soft") for value, provenance in soft)
    return tuple(
        GeneratedIntentValue(value, slot, provenance, position)
        for position, (value, provenance, slot) in enumerate(selected)
    )


class SimulatorIntentIndex:
    """Compact, label-free signatures reconstructed from public catalog fields."""

    def __init__(self, catalog_path: str | Path) -> None:
        self._signatures: dict[str, tuple[GeneratedIntentValue, ...]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                self._signatures[str(product["parent_asin"])] = generate_intent_signature(product)

    def signature(self, parent_asin: str) -> tuple[GeneratedIntentValue, ...]:
        return self._signatures.get(parent_asin, ())

    def score(
        self,
        parent_asin: str,
        evidence: Iterable[EvidenceLike],
    ) -> SimulatorLikelihood:
        signature = self.signature(parent_asin)
        by_value: dict[str, list[GeneratedIntentValue]] = {}
        for value in signature:
            by_value.setdefault(_normalized(value.value), []).append(value)

        active: list[EvidenceLike] = []
        seen: set[str] = set()
        for item in evidence:
            if item.source in {"category", "exclusion"}:
                continue
            normalized = _normalized(item.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            active.append(item)

        phrase_matches = 0
        slot_matches = 0
        provenance_matches = 0
        contradictions = 0
        positions: list[int] = []
        promoted = {
            value.provenance: _normalized(value.value).removeprefix("color: ")
            for value in signature
            if value.provenance in {"material", "color"}
        }
        for evidence_position, item in enumerate(active):
            normalized = _normalized(item.text)
            matches = by_value.get(normalized, [])
            if matches:
                phrase_matches += 1
                match = matches[0]
                positions.append(match.position)
                expected_slot = (
                    "soft" if item.source == "initial_preference"
                    else "hard" if item.source in {"hard_constraint", "override"}
                    else "hard" if evidence_position < 2
                    else "soft"
                )
                slot_matches += int(match.slot == expected_slot)
                attribute = item.attribute
                expected_provenance = (
                    attribute if attribute in {"material", "color", "budget"}
                    else None
                )
                provenance_matches += int(
                    expected_provenance is None or match.provenance == expected_provenance
                )
            attribute = item.attribute
            if attribute in {"material", "color"} and attribute in promoted:
                disclosed = normalized.removeprefix("color: ")
                # Only atomic facet disclosures are contradictions. Composite
                # feature phrases can legitimately mention several materials.
                pattern = MATERIAL_RE if attribute == "material" else COLOR_RE
                found = [
                    match.group(1).casefold()
                    for match in pattern.finditer(disclosed)
                ]
                if len(found) == 1 and found[0] != promoted[attribute]:
                    contradictions += 1

        ordered_pairs = sum(
            positions[left] < positions[right]
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        )
        return SimulatorLikelihood(
            contradictions=contradictions,
            phrase_matches=phrase_matches,
            slot_matches=slot_matches,
            provenance_matches=provenance_matches,
            ordered_pairs=ordered_pairs,
            missing_values=len(active) - phrase_matches,
        )
