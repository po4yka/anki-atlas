"""Heuristic quality scoring for flashcards.

Implements a 5-dimension rubric: clarity, atomicity, testability,
memorability, accuracy.  Each dimension scores 0.0-1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VAGUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^explain\s+",
        r"^describe\s+",
        r"^tell\s+(me\s+)?about\s+",
        r"^discuss\s+",
        r"^elaborate\s+(on\s+)?",
    )
)

_YES_NO_STARTERS: tuple[str, ...] = (
    "is ", "are ", "does ", "do ", "can ", "will ", "has ", "have ",
    "was ", "were ", "did ", "could ", "would ", "should ",
)

_MULTI_CONCEPT_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)
_ENUM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class QualityScore:
    """Five-dimension quality assessment of a flashcard."""

    clarity: float
    atomicity: float
    testability: float
    memorability: float
    accuracy: float

    @property
    def overall(self) -> float:
        """Average across all dimensions."""
        return (
            self.clarity + self.atomicity + self.testability
            + self.memorability + self.accuracy
        ) / 5.0


def assess_quality(*, front: str, back: str) -> QualityScore:
    """Score a card using heuristic checks.

    Each dimension is scored 0.0-1.0:
    - clarity: no vague words, has question structure, not yes/no
    - atomicity: short question, no multi-concept splitting
    - testability: answer is concrete and verifiable
    - memorability: reasonable length, no huge enumerations
    - accuracy: well-formed structure (question mark, non-empty)
    """
    front_lower = front.strip().lower()
    back_stripped = back.strip()

    clarity = _score_clarity(front_lower)
    atomicity = _score_atomicity(front_lower)
    testability = _score_testability(back_stripped)
    memorability = _score_memorability(back_stripped)
    accuracy = _score_accuracy(front.strip(), back_stripped)

    return QualityScore(
        clarity=clarity,
        atomicity=atomicity,
        testability=testability,
        memorability=memorability,
        accuracy=accuracy,
    )


def _score_clarity(front_lower: str) -> float:
    score = 1.0

    for pattern in _VAGUE_PATTERNS:
        if pattern.search(front_lower):
            score -= 0.4
            break

    if any(front_lower.startswith(s) for s in _YES_NO_STARTERS):
        score -= 0.3

    if "?" not in front_lower:
        score -= 0.2

    return max(0.0, score)


def _score_atomicity(front_lower: str) -> float:
    score = 1.0

    word_count = len(front_lower.split())
    if word_count > 30:
        score -= 0.4
    elif word_count > 20:
        score -= 0.2

    matches = _MULTI_CONCEPT_RE.findall(front_lower)
    if len(matches) >= 2:
        score -= 0.4
    elif len(matches) == 1:
        score -= 0.1

    return max(0.0, score)


def _score_testability(back: str) -> float:
    score = 1.0

    word_count = len(back.split())
    if word_count > 200:
        score -= 0.5
    elif word_count > 100:
        score -= 0.3

    if not back:
        return 0.0

    return max(0.0, score)


def _score_memorability(back: str) -> float:
    score = 1.0

    enum_items = len(_ENUM_RE.findall(back))
    if enum_items > 7:
        score -= 0.5
    elif enum_items > 4:
        score -= 0.2

    word_count = len(back.split())
    if word_count > 150:
        score -= 0.3

    return max(0.0, score)


def _score_accuracy(front: str, back: str) -> float:
    score = 1.0

    if not front:
        score -= 0.5
    if not back:
        score -= 0.5

    if front and "?" not in front:
        score -= 0.2

    return max(0.0, score)
