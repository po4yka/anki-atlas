"""Main card generation agent using PydanticAI for structured output."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

from packages.common.exceptions import CardGenerationError
from packages.generator.agents.models import (
    GeneratedCard,
    GenerationDeps,
    GenerationResult,
)
from packages.generator.prompts import card_generation_system, card_generation_user

log = structlog.get_logger()


class GeneratorAgent:
    """PydanticAI-based card generation agent."""

    def __init__(self, model_name: str, *, temperature: float = 0.3) -> None:
        self._model_name = model_name
        self._temperature = temperature

    async def generate(
        self,
        deps: GenerationDeps,
        qa_pairs: Sequence[tuple[str, str]],
    ) -> GenerationResult:
        """Generate cards from Q/A pairs using LLM structured output.

        Args:
            deps: Generation context (title, topic, etc.).
            qa_pairs: Sequence of (question, answer) tuples.

        Returns:
            GenerationResult with generated cards.

        Raises:
            CardGenerationError: If generation fails.
        """
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent
        except ImportError as exc:
            raise CardGenerationError(
                "pydantic-ai-slim is required for card generation. "
                "Install with: uv pip install 'anki-atlas[llm]'"
            ) from exc

        class _CardOutput(BaseModel):
            card_index: int = Field(ge=1)
            slug: str = Field(min_length=1)
            lang: str
            apf_html: str = Field(min_length=1)
            confidence: float = Field(ge=0.0, le=1.0, default=0.5)

        class _CardGenerationOutput(BaseModel):
            cards: list[_CardOutput]
            generation_notes: str = ""

        qa_text = "\n".join(f"Q{i}: {q}\nA{i}: {a}" for i, (q, a) in enumerate(qa_pairs, 1))
        language_tags_str = ", ".join(deps.language_tags) if deps.language_tags else "en"

        system_prompt = card_generation_system()
        user_prompt = card_generation_user(
            note_title=deps.note_title,
            topic=deps.topic,
            language_tags=language_tags_str,
            source_file=deps.source_file,
            qa_pairs=qa_text,
        )

        agent: Agent[None, _CardGenerationOutput] = Agent(
            self._model_name,
            output_type=_CardGenerationOutput,
            system_prompt=system_prompt,
            model_settings={"temperature": self._temperature},
        )

        start = time.monotonic()
        try:
            result = await agent.run(user_prompt)
        except Exception as exc:
            raise CardGenerationError(
                f"LLM generation failed: {exc}",
                context={"model": self._model_name},
            ) from exc
        elapsed = time.monotonic() - start

        cards = tuple(
            GeneratedCard(
                card_index=c.card_index,
                slug=c.slug,
                lang=c.lang,
                apf_html=c.apf_html,
                confidence=c.confidence,
                content_hash=hashlib.sha256(c.apf_html.encode()).hexdigest()[:16],
            )
            for c in result.output.cards
        )

        log.info(
            "cards_generated",
            count=len(cards),
            model=self._model_name,
            elapsed=round(elapsed, 2),
        )

        return GenerationResult(
            cards=cards,
            total_cards=len(cards),
            model_used=self._model_name,
            generation_time=round(elapsed, 3),
        )
