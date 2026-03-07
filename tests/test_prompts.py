"""Tests for packages.generator.prompts."""

from __future__ import annotations

import pytest

from packages.common.exceptions import ConfigurationError
from packages.generator.prompts.enhancement import (
    cloze_conversion_system,
    cloze_conversion_user,
    enhancement_system,
    enhancement_user,
    mnemonic_generation_system,
    mnemonic_generation_user,
    reverse_generation_system,
    reverse_generation_user,
    split_suggestion_system,
    split_suggestion_user,
)
from packages.generator.prompts.generation import (
    card_generation_system,
    card_generation_user,
    card_splitting_system,
    card_splitting_user,
    context_enrichment_system,
    context_enrichment_user,
)
from packages.generator.prompts.templates import (
    PromptTemplate,
    create_simple_template,
    merge_templates,
    parse_template_string,
    validate_template,
)
from packages.generator.prompts.validation import (
    memorization_assessment_system,
    memorization_assessment_user,
    post_validation_system,
    post_validation_user,
    pre_validation_system,
    pre_validation_user,
)

# ===========================================================================
# PromptTemplate
# ===========================================================================


class TestPromptTemplate:
    def test_frozen(self) -> None:
        t = PromptTemplate(prompt_body="hello")
        with pytest.raises(AttributeError):
            t.prompt_body = "changed"  # type: ignore[misc]

    def test_substitute(self) -> None:
        t = PromptTemplate(prompt_body="Hello {name}, you have {count} items.")
        result = t.substitute(name="Alice", count=3)
        assert result == "Hello Alice, you have 3 items."

    def test_substitute_none_value(self) -> None:
        t = PromptTemplate(prompt_body="Hello {name}!")
        result = t.substitute(name=None)
        assert result == "Hello !"

    def test_substitute_missing_key_leaves_placeholder(self) -> None:
        t = PromptTemplate(prompt_body="Hello {name}!")
        result = t.substitute(other="x")
        assert result == "Hello {name}!"

    def test_get_required_variables(self) -> None:
        t = PromptTemplate(prompt_body="{a} and {b} and {a}")
        variables = t.get_required_variables()
        assert set(variables) == {"a", "b"}

    def test_get_required_variables_empty(self) -> None:
        t = PromptTemplate(prompt_body="no variables here")
        assert t.get_required_variables() == ()

    def test_defaults(self) -> None:
        t = PromptTemplate()
        assert t.deck is None
        assert t.note_type is None
        assert t.field_map is None
        assert t.quality_check is None
        assert t.prompt_body == ""


# ===========================================================================
# parse_template_string
# ===========================================================================


class TestParseTemplateString:
    def test_with_frontmatter(self) -> None:
        content = """\
---
deck: "My Deck"
note_type: "Basic"
field_map:
  front: "Front"
  back: "Back"
---
Generate a card for {topic}."""
        template = parse_template_string(content)
        assert template.deck == "My Deck"
        assert template.note_type == "Basic"
        assert template.field_map == {"front": "Front", "back": "Back"}
        assert "Generate a card for {topic}" in template.prompt_body

    def test_without_frontmatter(self) -> None:
        content = "Just a plain prompt body."
        template = parse_template_string(content)
        assert template.deck is None
        assert template.prompt_body == "Just a plain prompt body."

    def test_invalid_yaml_raises(self) -> None:
        content = "---\n: invalid: yaml:\n---\nbody"
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            parse_template_string(content)

    def test_empty_frontmatter(self) -> None:
        content = "---\n\n---\nBody text here"
        template = parse_template_string(content)
        assert template.deck is None
        assert template.prompt_body == "Body text here"


# ===========================================================================
# validate_template
# ===========================================================================


class TestValidateTemplate:
    def test_valid_template(self) -> None:
        t = PromptTemplate(
            prompt_body="This is a long enough prompt body that passes the 50 char check."
        )
        errors = validate_template(t)
        assert errors == ()

    def test_empty_body(self) -> None:
        t = PromptTemplate(prompt_body="")
        errors = validate_template(t)
        assert any("empty" in e.lower() for e in errors)

    def test_short_body(self) -> None:
        t = PromptTemplate(prompt_body="Short")
        errors = validate_template(t)
        assert any("too short" in e.lower() for e in errors)

    def test_unbalanced_braces(self) -> None:
        t = PromptTemplate(
            prompt_body="This has an {unclosed brace but enough chars to pass the check"
        )
        errors = validate_template(t)
        assert any("unbalanced" in e.lower() for e in errors)

    def test_empty_field_map(self) -> None:
        t = PromptTemplate(
            prompt_body="x" * 60,
            field_map={},
        )
        errors = validate_template(t)
        assert any("field_map" in e.lower() for e in errors)


# ===========================================================================
# create_simple_template / merge_templates
# ===========================================================================


class TestConvenience:
    def test_create_simple(self) -> None:
        t = create_simple_template("Generate card for {topic}", deck="My Deck")
        assert t.deck == "My Deck"
        assert t.note_type is None
        assert "{topic}" in t.prompt_body

    def test_merge_override_wins(self) -> None:
        base = PromptTemplate(deck="Base Deck", prompt_body="base body")
        override = PromptTemplate(deck="Override Deck", prompt_body="override body")
        merged = merge_templates(base, override)
        assert merged.deck == "Override Deck"
        assert merged.prompt_body == "override body"

    def test_merge_fallback_to_base(self) -> None:
        base = PromptTemplate(deck="Base Deck", note_type="Basic", prompt_body="body")
        override = PromptTemplate()
        merged = merge_templates(base, override)
        assert merged.deck == "Base Deck"
        assert merged.note_type == "Basic"
        assert merged.prompt_body == "body"

    def test_merge_field_maps(self) -> None:
        base = PromptTemplate(field_map={"a": "1"}, prompt_body="body")
        override = PromptTemplate(field_map={"b": "2"}, prompt_body="")
        merged = merge_templates(base, override)
        assert merged.field_map == {"a": "1", "b": "2"}


# ===========================================================================
# Generation prompts
# ===========================================================================


class TestGenerationPrompts:
    def test_card_generation_system_returns_string(self) -> None:
        result = card_generation_system()
        assert "card generation" in result.lower()
        assert "JSON" in result

    def test_card_generation_user_formats_params(self) -> None:
        result = card_generation_user(
            note_title="Test",
            topic="Python",
            language_tags="en",
            source_file="test.md",
            qa_pairs="Q: What? A: That.",
        )
        assert "Test" in result
        assert "Python" in result
        assert "Q: What?" in result

    def test_card_splitting_system(self) -> None:
        result = card_splitting_system()
        assert "split" in result.lower()

    def test_card_splitting_user(self) -> None:
        result = card_splitting_user(
            title="Test", topic="Python", language_tags="en", content="Some content"
        )
        assert "Test" in result
        assert "Some content" in result

    def test_context_enrichment_system(self) -> None:
        result = context_enrichment_system()
        assert "context" in result.lower()
        assert "wikilink" in result.lower()

    def test_context_enrichment_user(self) -> None:
        result = context_enrichment_user(
            note_title="T",
            topic="P",
            note_content="NC",
            question="Q",
            answer="A",
            linked_notes="LN",
        )
        assert "Q" in result
        assert "LN" in result


# ===========================================================================
# Validation prompts
# ===========================================================================


class TestValidationPrompts:
    def test_pre_validation_system(self) -> None:
        result = pre_validation_system()
        assert "pre-validation" in result.lower()

    def test_pre_validation_user(self) -> None:
        result = pre_validation_user(
            title="T",
            topic="P",
            tags="t1",
            language_tags="en",
            qa_count=3,
            content_preview="preview",
        )
        assert "T" in result
        assert "3" in result

    def test_post_validation_system(self) -> None:
        result = post_validation_system()
        assert "APF" in result

    def test_post_validation_user(self) -> None:
        result = post_validation_user(
            source_note="note.md",
            expected_lang="en",
            apf_content="<!-- BEGIN_CARDS -->",
        )
        assert "note.md" in result
        assert "BEGIN_CARDS" in result

    def test_memorization_assessment_system(self) -> None:
        result = memorization_assessment_system()
        assert "memorization" in result.lower()
        assert "atomicity" in result.lower()

    def test_memorization_assessment_user(self) -> None:
        result = memorization_assessment_user(question="What is X?", answer="Y", context="ctx")
        assert "What is X?" in result
        assert "ctx" in result


# ===========================================================================
# Enhancement prompts
# ===========================================================================


class TestEnhancementPrompts:
    def test_enhancement_system(self) -> None:
        result = enhancement_system()
        assert "enhancement" in result.lower()

    def test_enhancement_user(self) -> None:
        result = enhancement_user(front="Q", back="A", card_type="Simple", tags="t", language="en")
        assert "Q" in result
        assert "Simple" in result

    def test_split_suggestion_system(self) -> None:
        result = split_suggestion_system()
        assert "atomicity" in result.lower()

    def test_split_suggestion_user(self) -> None:
        result = split_suggestion_user(front="F", back="B")
        assert "F" in result

    def test_reverse_generation_system(self) -> None:
        result = reverse_generation_system()
        assert "reversed" in result.lower()

    def test_reverse_generation_user(self) -> None:
        result = reverse_generation_user(front="F", back="B")
        assert "F" in result

    def test_cloze_conversion_system(self) -> None:
        result = cloze_conversion_system()
        assert "cloze" in result.lower()

    def test_cloze_conversion_user(self) -> None:
        result = cloze_conversion_user(front="F", back="B")
        assert "F" in result

    def test_mnemonic_generation_system(self) -> None:
        result = mnemonic_generation_system()
        assert "mnemonic" in result.lower()

    def test_mnemonic_generation_user(self) -> None:
        result = mnemonic_generation_user(front="F", back="B")
        assert "F" in result


# ===========================================================================
# Import from top-level package
# ===========================================================================


class TestPackageExports:
    def test_import_from_prompts_package(self) -> None:
        from packages.generator.prompts import (
            PromptTemplate,
            card_generation_system,
            enhancement_system,
            memorization_assessment_system,
            parse_template_string,
            pre_validation_system,
        )

        assert PromptTemplate is not None
        assert callable(card_generation_system)
        assert callable(enhancement_system)
        assert callable(memorization_assessment_system)
        assert callable(parse_template_string)
        assert callable(pre_validation_system)
