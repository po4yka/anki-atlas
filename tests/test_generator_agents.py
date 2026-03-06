"""Tests for packages.generator.agents."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.common.exceptions import CardGenerationError
from packages.generator.agents.models import (
    GeneratedCard,
    GenerationDeps,
    GenerationResult,
    SplitDecision,
    SplitPlan,
)

# ---------------------------------------------------------------------------
# Mock pydantic_ai fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pydantic_ai() -> Any:
    """Inject a fake pydantic_ai module so lazy imports succeed."""
    mod = ModuleType("pydantic_ai")
    mock_agent_cls = MagicMock()
    mod.Agent = mock_agent_cls  # type: ignore[attr-defined]
    old = sys.modules.get("pydantic_ai")
    sys.modules["pydantic_ai"] = mod
    yield mock_agent_cls
    if old is None:
        sys.modules.pop("pydantic_ai", None)
    else:
        sys.modules["pydantic_ai"] = old


def _make_mock_agent_run(output: Any) -> MagicMock:
    """Create a mock Agent instance whose run() returns output."""
    mock_result = MagicMock()
    mock_result.output = output
    mock_instance = MagicMock()
    mock_instance.run = AsyncMock(return_value=mock_result)
    return mock_instance


# ---------------------------------------------------------------------------
# GeneratedCard
# ---------------------------------------------------------------------------


class TestGeneratedCard:
    def test_creation(self) -> None:
        card = GeneratedCard(
            card_index=1,
            slug="test-card-1-en",
            lang="en",
            apf_html="<div>test</div>",
            confidence=0.9,
            content_hash="abc123",
        )
        assert card.card_index == 1
        assert card.slug == "test-card-1-en"
        assert card.lang == "en"
        assert card.apf_html == "<div>test</div>"
        assert card.confidence == 0.9
        assert card.content_hash == "abc123"

    def test_defaults(self) -> None:
        card = GeneratedCard(card_index=1, slug="s", lang="en", apf_html="x")
        assert card.confidence == 0.0
        assert card.content_hash == ""

    def test_frozen(self) -> None:
        card = GeneratedCard(card_index=1, slug="s", lang="en", apf_html="x")
        with pytest.raises(FrozenInstanceError):
            card.slug = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_creation(self) -> None:
        card = GeneratedCard(card_index=1, slug="s", lang="en", apf_html="x")
        result = GenerationResult(
            cards=(card,),
            total_cards=1,
            model_used="openai:gpt-4o",
            generation_time=1.5,
            warnings=("minor issue",),
        )
        assert result.cards == (card,)
        assert result.total_cards == 1
        assert result.model_used == "openai:gpt-4o"
        assert result.generation_time == 1.5
        assert result.warnings == ("minor issue",)

    def test_defaults(self) -> None:
        result = GenerationResult(cards=(), total_cards=0, model_used="test")
        assert result.generation_time == 0.0
        assert result.warnings == ()

    def test_frozen(self) -> None:
        result = GenerationResult(cards=(), total_cards=0, model_used="test")
        with pytest.raises(FrozenInstanceError):
            result.model_used = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GenerationDeps
# ---------------------------------------------------------------------------


class TestGenerationDeps:
    def test_creation(self) -> None:
        deps = GenerationDeps(
            note_title="Test Note",
            topic="Python",
            language_tags=("en", "ru"),
            source_file="notes/test.md",
        )
        assert deps.note_title == "Test Note"
        assert deps.topic == "Python"
        assert deps.language_tags == ("en", "ru")
        assert deps.source_file == "notes/test.md"

    def test_defaults(self) -> None:
        deps = GenerationDeps(note_title="T", topic="P")
        assert deps.language_tags == ()
        assert deps.source_file == ""

    def test_frozen(self) -> None:
        deps = GenerationDeps(note_title="T", topic="P")
        with pytest.raises(FrozenInstanceError):
            deps.topic = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SplitPlan / SplitDecision
# ---------------------------------------------------------------------------


class TestSplitPlan:
    def test_creation(self) -> None:
        plan = SplitPlan(
            card_number=1,
            concept="variables",
            question="What are variables?",
            answer_summary="Named storage locations",
        )
        assert plan.card_number == 1
        assert plan.concept == "variables"

    def test_frozen(self) -> None:
        plan = SplitPlan(card_number=1, concept="c", question="q", answer_summary="a")
        with pytest.raises(FrozenInstanceError):
            plan.concept = "x"  # type: ignore[misc]


class TestSplitDecision:
    def test_creation(self) -> None:
        plan = SplitPlan(card_number=1, concept="c", question="q", answer_summary="a")
        decision = SplitDecision(
            should_split=True,
            card_count=2,
            plans=(plan,),
            reasoning="Multiple concepts",
        )
        assert decision.should_split is True
        assert decision.card_count == 2
        assert len(decision.plans) == 1
        assert decision.reasoning == "Multiple concepts"

    def test_defaults(self) -> None:
        decision = SplitDecision(should_split=False, card_count=1)
        assert decision.plans == ()
        assert decision.reasoning == ""

    def test_frozen(self) -> None:
        decision = SplitDecision(should_split=False, card_count=1)
        with pytest.raises(FrozenInstanceError):
            decision.should_split = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GeneratorAgent
# ---------------------------------------------------------------------------


class TestGeneratorAgent:
    def test_init(self) -> None:
        from packages.generator.agents.generator import GeneratorAgent

        agent = GeneratorAgent("openai:gpt-4o", temperature=0.5)
        assert agent._model_name == "openai:gpt-4o"
        assert agent._temperature == 0.5

    async def test_generate_with_mock(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.generator import GeneratorAgent

        mock_card = MagicMock()
        mock_card.card_index = 1
        mock_card.slug = "python-var-1-en"
        mock_card.lang = "en"
        mock_card.apf_html = "<div>What are variables?</div>"
        mock_card.confidence = 0.9

        mock_output = MagicMock()
        mock_output.cards = [mock_card]
        mock_output.generation_notes = ""

        mock_agent_instance = _make_mock_agent_run(mock_output)
        mock_pydantic_ai.return_value = mock_agent_instance

        deps = GenerationDeps(note_title="Variables", topic="Python")
        qa_pairs = [("What are variables?", "Named storage locations")]

        agent = GeneratorAgent("openai:gpt-4o")
        result = await agent.generate(deps, qa_pairs)

        assert isinstance(result, GenerationResult)
        assert result.total_cards == 1
        assert result.cards[0].slug == "python-var-1-en"
        assert result.cards[0].confidence == 0.9
        assert result.cards[0].content_hash != ""
        assert result.model_used == "openai:gpt-4o"
        assert result.generation_time >= 0

    async def test_generate_import_error(self) -> None:
        from packages.generator.agents.generator import GeneratorAgent

        deps = GenerationDeps(note_title="T", topic="P")

        with (
            patch.dict(sys.modules, {"pydantic_ai": None, "pydantic": None}),
            pytest.raises(CardGenerationError),
        ):
            agent = GeneratorAgent("test")
            await agent.generate(deps, [("q", "a")])


# ---------------------------------------------------------------------------
# PreValidatorAgent
# ---------------------------------------------------------------------------


class TestPreValidatorAgent:
    def test_init(self) -> None:
        from packages.generator.agents.validator import PreValidatorAgent

        agent = PreValidatorAgent("openai:gpt-4o")
        assert agent._model_name == "openai:gpt-4o"
        assert agent._temperature == 0.0

    async def test_validate_valid(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.validator import PreValidatorAgent

        mock_output = MagicMock()
        mock_output.is_valid = True
        mock_output.error_type = "none"
        mock_output.error_details = ""
        mock_output.confidence = 0.95

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="Test", topic="Python")

        agent = PreValidatorAgent("openai:gpt-4o")
        result = await agent.validate("Some content", deps, qa_count=3)

        assert result.is_valid

    async def test_validate_invalid(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.validator import PreValidatorAgent

        mock_output = MagicMock()
        mock_output.is_valid = False
        mock_output.error_type = "frontmatter"
        mock_output.error_details = "Missing title"
        mock_output.confidence = 0.9

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="", topic="Python")

        agent = PreValidatorAgent("openai:gpt-4o")
        result = await agent.validate("content", deps)

        assert not result.is_valid
        assert any("Missing title" in i.message for i in result.issues)


# ---------------------------------------------------------------------------
# PostValidatorAgent
# ---------------------------------------------------------------------------


class TestPostValidatorAgent:
    async def test_validate_valid(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.validator import PostValidatorAgent

        mock_output = MagicMock()
        mock_output.is_valid = True
        mock_output.error_type = "none"
        mock_output.error_details = ""
        mock_output.issues = []
        mock_output.confidence = 0.95

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="Test", topic="Python")
        cards = (GeneratedCard(card_index=1, slug="s", lang="en", apf_html="<div>test</div>"),)

        agent = PostValidatorAgent("openai:gpt-4o")
        result = await agent.validate(cards, deps)

        assert result.is_valid

    async def test_validate_with_issues(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.validator import PostValidatorAgent

        mock_output = MagicMock()
        mock_output.is_valid = False
        mock_output.error_type = "syntax"
        mock_output.error_details = "Unclosed tag"
        mock_output.issues = ["Minor formatting issue"]
        mock_output.confidence = 0.7

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="Test", topic="Python")
        cards = (GeneratedCard(card_index=1, slug="s", lang="en", apf_html="<div>bad"),)

        agent = PostValidatorAgent("openai:gpt-4o")
        result = await agent.validate(cards, deps)

        assert not result.is_valid
        assert len(result.issues) == 2


# ---------------------------------------------------------------------------
# EnhancerAgent
# ---------------------------------------------------------------------------


class TestEnhancerAgent:
    def test_init(self) -> None:
        from packages.generator.agents.enhancer import EnhancerAgent

        agent = EnhancerAgent("openai:gpt-4o", temperature=0.5)
        assert agent._model_name == "openai:gpt-4o"
        assert agent._temperature == 0.5

    async def test_enhance_with_improvements(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.enhancer import EnhancerAgent

        mock_output = MagicMock()
        mock_output.enhanced_front = "<div>Improved question</div>"
        mock_output.enhanced_back = "<div>Improved answer</div>"
        mock_output.improvements = ["Clarified wording"]
        mock_output.confidence = 0.85

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        card = GeneratedCard(
            card_index=1, slug="test-1-en", lang="en", apf_html="<div>original</div>"
        )
        deps = GenerationDeps(note_title="Test", topic="Python")

        agent = EnhancerAgent("openai:gpt-4o")
        enhanced = await agent.enhance(card, deps)

        assert enhanced.apf_html == "<div>Improved question</div>"
        assert enhanced.confidence == 0.85
        assert enhanced.slug == card.slug

    async def test_enhance_no_improvements(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.enhancer import EnhancerAgent

        mock_output = MagicMock()
        mock_output.enhanced_front = "<div>original</div>"
        mock_output.enhanced_back = ""
        mock_output.improvements = []
        mock_output.confidence = 0.9

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        card = GeneratedCard(
            card_index=1, slug="test-1-en", lang="en", apf_html="<div>original</div>"
        )
        deps = GenerationDeps(note_title="Test", topic="Python")

        agent = EnhancerAgent("openai:gpt-4o")
        enhanced = await agent.enhance(card, deps)

        assert enhanced is card

    async def test_suggest_split(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.enhancer import EnhancerAgent

        mock_plan = MagicMock()
        mock_plan.card_number = 1
        mock_plan.concept = "variables"
        mock_plan.question = "What are variables?"
        mock_plan.answer_summary = "Named storage"

        mock_output = MagicMock()
        mock_output.should_split = True
        mock_output.card_count = 2
        mock_output.plans = [mock_plan]
        mock_output.reasoning = "Two concepts"

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="Test", topic="Python")

        agent = EnhancerAgent("openai:gpt-4o")
        decision = await agent.suggest_split("content about vars and funcs", deps)

        assert isinstance(decision, SplitDecision)
        assert decision.should_split is True
        assert decision.card_count == 2
        assert len(decision.plans) == 1
        assert decision.plans[0].concept == "variables"

    async def test_suggest_no_split(self, mock_pydantic_ai: Any) -> None:
        from packages.generator.agents.enhancer import EnhancerAgent

        mock_output = MagicMock()
        mock_output.should_split = False
        mock_output.card_count = 1
        mock_output.plans = []
        mock_output.reasoning = "Single concept"

        mock_pydantic_ai.return_value = _make_mock_agent_run(mock_output)
        deps = GenerationDeps(note_title="Test", topic="Python")

        agent = EnhancerAgent("openai:gpt-4o")
        decision = await agent.suggest_split("simple content", deps)

        assert decision.should_split is False
        assert decision.card_count == 1
        assert decision.plans == ()


# ---------------------------------------------------------------------------
# Import re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_import_from_package(self) -> None:
        from packages.generator.agents import (
            EnhancerAgent,
            GeneratedCard,
            GenerationDeps,
            GenerationResult,
            GeneratorAgent,
            PostValidatorAgent,
            PreValidatorAgent,
            SplitDecision,
            SplitPlan,
        )

        assert GeneratedCard is not None
        assert GenerationResult is not None
        assert GenerationDeps is not None
        assert GeneratorAgent is not None
        assert PreValidatorAgent is not None
        assert PostValidatorAgent is not None
        assert EnhancerAgent is not None
        assert SplitDecision is not None
        assert SplitPlan is not None
