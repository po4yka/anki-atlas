"""Enhancement agent for improving and splitting cards."""

from __future__ import annotations

import structlog

from packages.common.exceptions import CardGenerationError
from packages.generator.agents.models import (
    GeneratedCard,
    GenerationDeps,
    SplitDecision,
    SplitPlan,
)
from packages.generator.prompts import (
    card_splitting_system,
    card_splitting_user,
    enhancement_system,
    enhancement_user,
)

log = structlog.get_logger()


class EnhancerAgent:
    """Agent for enhancing cards and suggesting splits."""

    def __init__(self, model_name: str, *, temperature: float = 0.3) -> None:
        self._model_name = model_name
        self._temperature = temperature

    async def enhance(
        self,
        card: GeneratedCard,
        deps: GenerationDeps,
    ) -> GeneratedCard:
        """Enhance a single card using LLM suggestions.

        Args:
            card: Card to enhance.
            deps: Generation context.

        Returns:
            Enhanced card (or original if no improvements suggested).

        Raises:
            CardGenerationError: If enhancement fails.
        """
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent
        except ImportError as exc:
            raise CardGenerationError(
                "pydantic-ai-slim is required for card enhancement. "
                "Install with: uv pip install 'anki-atlas[llm]'"
            ) from exc

        class _EnhancementOutput(BaseModel):
            enhanced_front: str
            enhanced_back: str
            improvements: list[str] = Field(default_factory=list)
            confidence: float = Field(ge=0.0, le=1.0, default=0.5)

        language = card.lang or "en"

        system_prompt = enhancement_system()
        user_prompt = enhancement_user(
            front=card.apf_html,
            back="",
            card_type="Simple",
            tags=deps.topic,
            language=language,
        )

        agent: Agent[None, _EnhancementOutput] = Agent(
            self._model_name,
            output_type=_EnhancementOutput,
            system_prompt=system_prompt,
            model_settings={"temperature": self._temperature},
        )

        try:
            result = await agent.run(user_prompt)
        except Exception as exc:
            raise CardGenerationError(
                f"Enhancement agent failed: {exc}",
                context={"model": self._model_name},
            ) from exc

        output = result.output

        if not output.improvements:
            log.debug("no_enhancements_suggested", slug=card.slug)
            return card

        log.info(
            "card_enhanced",
            slug=card.slug,
            improvements=output.improvements,
        )

        return GeneratedCard(
            card_index=card.card_index,
            slug=card.slug,
            lang=card.lang,
            apf_html=output.enhanced_front,
            confidence=output.confidence,
            content_hash=card.content_hash,
        )

    async def suggest_split(
        self,
        content: str,
        deps: GenerationDeps,
    ) -> SplitDecision:
        """Analyze content and suggest whether to split into multiple cards.

        Args:
            content: Note content to analyze.
            deps: Generation context.

        Returns:
            SplitDecision with split recommendation.

        Raises:
            CardGenerationError: If split analysis fails.
        """
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent
        except ImportError as exc:
            raise CardGenerationError(
                "pydantic-ai-slim is required for split analysis. "
                "Install with: uv pip install 'anki-atlas[llm]'"
            ) from exc

        class _SplitPlanItem(BaseModel):
            card_number: int = Field(ge=1)
            concept: str
            question: str
            answer_summary: str

        class _SplitDecisionOutput(BaseModel):
            should_split: bool
            card_count: int = Field(ge=1, default=1)
            plans: list[_SplitPlanItem] = Field(default_factory=list)
            reasoning: str = ""

        language_tags_str = ", ".join(deps.language_tags) if deps.language_tags else "en"

        system_prompt = card_splitting_system()
        user_prompt = card_splitting_user(
            title=deps.note_title,
            topic=deps.topic,
            language_tags=language_tags_str,
            content=content,
        )

        agent: Agent[None, _SplitDecisionOutput] = Agent(
            self._model_name,
            output_type=_SplitDecisionOutput,
            system_prompt=system_prompt,
            model_settings={"temperature": self._temperature},
        )

        try:
            result = await agent.run(user_prompt)
        except Exception as exc:
            raise CardGenerationError(
                f"Split analysis failed: {exc}",
                context={"model": self._model_name},
            ) from exc

        output = result.output

        plans = tuple(
            SplitPlan(
                card_number=p.card_number,
                concept=p.concept,
                question=p.question,
                answer_summary=p.answer_summary,
            )
            for p in output.plans
        )

        log.info(
            "split_analysis_complete",
            should_split=output.should_split,
            card_count=output.card_count,
        )

        return SplitDecision(
            should_split=output.should_split,
            card_count=output.card_count,
            plans=plans,
            reasoning=output.reasoning,
        )
