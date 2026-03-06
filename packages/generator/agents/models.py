"""Domain models for card generation agents.

Frozen dataclasses for GeneratedCard, GenerationResult, and GenerationDeps.
These are the public API; PydanticAI output schemas are internal to each agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GeneratedCard:
    """A single generated flashcard."""

    card_index: int
    slug: str
    lang: str
    apf_html: str
    confidence: float = 0.0
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of a card generation run."""

    cards: tuple[GeneratedCard, ...]
    total_cards: int
    model_used: str
    generation_time: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationDeps:
    """Dependencies passed to generation agents."""

    note_title: str
    topic: str
    language_tags: tuple[str, ...] = ()
    source_file: str = ""


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """A single planned card from a split decision."""

    card_number: int
    concept: str
    question: str
    answer_summary: str


@dataclass(frozen=True, slots=True)
class SplitDecision:
    """Result of a card splitting analysis."""

    should_split: bool
    card_count: int
    plans: tuple[SplitPlan, ...] = field(default_factory=tuple)
    reasoning: str = ""
