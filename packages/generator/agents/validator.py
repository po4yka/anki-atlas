"""Pre and post validation agents using PydanticAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from packages.common.exceptions import CardValidationError
from packages.generator.prompts import (
    post_validation_system,
    post_validation_user,
    pre_validation_system,
    pre_validation_user,
)
from packages.validation import Severity, ValidationIssue, ValidationResult

if TYPE_CHECKING:
    from packages.generator.agents.models import GeneratedCard, GenerationDeps

log = structlog.get_logger()


class PreValidatorAgent:
    """Validates note structure before card generation."""

    def __init__(self, model_name: str, *, temperature: float = 0.0) -> None:
        self._model_name = model_name
        self._temperature = temperature

    async def validate(
        self,
        content: str,
        deps: GenerationDeps,
        *,
        qa_count: int = 0,
    ) -> ValidationResult:
        """Validate note structure before generation.

        Args:
            content: Note content preview.
            deps: Generation context.
            qa_count: Number of Q/A pairs found by parser.

        Returns:
            ValidationResult with any issues found.

        Raises:
            CardValidationError: If validation agent fails.
        """
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent
        except ImportError as exc:
            raise CardValidationError(
                "pydantic-ai-slim is required for LLM validation. "
                "Install with: uv pip install 'anki-atlas[llm]'"
            ) from exc

        class _PreValidationOutput(BaseModel):
            is_valid: bool
            error_type: str = "none"
            error_details: str = ""
            confidence: float = Field(ge=0.0, le=1.0, default=0.5)

        language_tags_str = ", ".join(deps.language_tags) if deps.language_tags else "en"

        system_prompt = pre_validation_system()
        user_prompt = pre_validation_user(
            title=deps.note_title,
            topic=deps.topic,
            tags="",
            language_tags=language_tags_str,
            qa_count=qa_count,
            content_preview=content,
        )

        agent: Agent[None, _PreValidationOutput] = Agent(
            self._model_name,
            output_type=_PreValidationOutput,
            system_prompt=system_prompt,
            model_settings={"temperature": self._temperature},
        )

        try:
            result = await agent.run(user_prompt)
        except Exception as exc:
            raise CardValidationError(
                f"Pre-validation agent failed: {exc}",
                context={"model": self._model_name},
            ) from exc

        output = result.output
        issues: list[ValidationIssue] = []
        if not output.is_valid:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    message=output.error_details or f"Pre-validation failed: {output.error_type}",
                    location=output.error_type,
                )
            )

        log.info(
            "pre_validation_complete",
            is_valid=output.is_valid,
            error_type=output.error_type,
        )

        return ValidationResult(issues=tuple(issues))


class PostValidatorAgent:
    """Validates generated cards after generation."""

    def __init__(self, model_name: str, *, temperature: float = 0.0) -> None:
        self._model_name = model_name
        self._temperature = temperature

    async def validate(
        self,
        cards: tuple[GeneratedCard, ...],
        deps: GenerationDeps,
    ) -> ValidationResult:
        """Validate generated cards.

        Args:
            cards: Generated cards to validate.
            deps: Generation context.

        Returns:
            ValidationResult with any issues found.

        Raises:
            CardValidationError: If validation agent fails.
        """
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent
        except ImportError as exc:
            raise CardValidationError(
                "pydantic-ai-slim is required for LLM validation. "
                "Install with: uv pip install 'anki-atlas[llm]'"
            ) from exc

        class _PostValidationOutput(BaseModel):
            is_valid: bool
            error_type: str = "none"
            error_details: str = ""
            issues: list[str] = Field(default_factory=list)
            confidence: float = Field(ge=0.0, le=1.0, default=0.5)

        apf_content = "\n---\n".join(c.apf_html for c in cards)
        language_tags_str = ", ".join(deps.language_tags) if deps.language_tags else "en"

        system_prompt = post_validation_system()
        user_prompt = post_validation_user(
            source_note=deps.note_title,
            expected_lang=language_tags_str,
            apf_content=apf_content,
        )

        agent: Agent[None, _PostValidationOutput] = Agent(
            self._model_name,
            output_type=_PostValidationOutput,
            system_prompt=system_prompt,
            model_settings={"temperature": self._temperature},
        )

        try:
            result = await agent.run(user_prompt)
        except Exception as exc:
            raise CardValidationError(
                f"Post-validation agent failed: {exc}",
                context={"model": self._model_name},
            ) from exc

        output = result.output
        issues: list[ValidationIssue] = []
        if not output.is_valid:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    message=output.error_details or f"Post-validation failed: {output.error_type}",
                    location=output.error_type,
                )
            )
        for issue_text in output.issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    message=issue_text,
                )
            )

        log.info(
            "post_validation_complete",
            is_valid=output.is_valid,
            issue_count=len(issues),
        )

        return ValidationResult(issues=tuple(issues))
