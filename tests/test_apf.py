"""Tests for packages.card.apf -- APF format modules."""

from __future__ import annotations

from types import SimpleNamespace

from packages.card.apf.converter import _basic_markdown_to_html, sanitize_html
from packages.card.apf.generator import GenerationResult, HTMLTemplateGenerator
from packages.card.apf.linter import LintResult, validate_apf
from packages.card.apf.renderer import APFRenderer, APFSentinelValidator
from packages.card.apf.validator import (
    MarkdownValidationResult,
    validate_card_html,
    validate_markdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    card_index: int = 1,
    slug: str = "test-card",
    slug_base: str = "test",
    card_type: str = "Simple",
    tags: tuple[str, ...] = ("python", "testing", "basics"),
    lang: str = "en",
    guid: str = "abc123",
    source_path: str = "",
    source_anchor: str = "",
    title: str = "Test Title",
    key_point_code: str = "",
    key_point_code_lang: str = "",
    key_point_notes: tuple[str, ...] = ("<li>Note 1</li>",),
    other_notes: str = "",
    extra: str = "",
) -> SimpleNamespace:
    front = SimpleNamespace(
        title=title,
        key_point_code=key_point_code,
        key_point_code_lang=key_point_code_lang,
        key_point_notes=list(key_point_notes),
        other_notes=other_notes,
        extra=extra,
    )
    return SimpleNamespace(
        card_index=card_index,
        slug=slug,
        slug_base=slug_base,
        card_type=card_type,
        tags=list(tags),
        lang=lang,
        guid=guid,
        source_path=source_path,
        source_anchor=source_anchor,
        front=front,
    )


def _make_valid_apf(
    slug: str = "test-card",
    card_type: str = "Simple",
    tags: str = "python testing basics",
) -> str:
    """Build a minimal valid APF HTML document."""
    import json

    tags_json = json.dumps(tags.split())
    manifest = (
        f'{{"slug":"{slug}","lang":"en","type":"{card_type}",'
        f'"tags":{tags_json},"guid":"abc123"}}'
    )
    return (
        "<!-- PROMPT_VERSION: apf-v2.1 -->\n"
        "<!-- BEGIN_CARDS -->\n"
        f"<!-- Card 1 | slug: {slug} | CardType: {card_type} | Tags: {tags} -->\n"
        "<!-- Title -->\nTest Title\n"
        "<!-- Key point (code block / image) -->\n"
        '<pre><code class="language-python">print("hi")</code></pre>\n'
        "<!-- Key point notes -->\n"
        "<ul>\n<li>Note 1</li>\n<li>Note 2</li>\n<li>Note 3</li>\n</ul>\n"
        f"<!-- manifest: {manifest} -->\n"
        "<!-- END_CARDS -->\n"
        "END_OF_CARDS"
    )


# ===================================================================
# APFRenderer
# ===================================================================


class TestAPFRenderer:
    def test_render_produces_all_sentinels(self) -> None:
        spec = _make_spec()
        renderer = APFRenderer()
        result = renderer.render(spec)

        assert "<!-- PROMPT_VERSION: apf-v2.1 -->" in result
        assert "<!-- BEGIN_CARDS -->" in result
        assert "<!-- END_CARDS -->" in result
        assert "END_OF_CARDS" in result
        assert "<!-- Title -->" in result
        assert "<!-- manifest:" in result

    def test_render_includes_slug_in_header(self) -> None:
        spec = _make_spec(slug="my-slug")
        result = APFRenderer().render(spec)
        assert "slug: my-slug" in result

    def test_render_code_block(self) -> None:
        spec = _make_spec(
            key_point_code='print("hello")',
            key_point_code_lang="python",
        )
        result = APFRenderer().render(spec)
        assert 'class="language-python"' in result
        assert "print(" in result

    def test_render_batch_empty(self) -> None:
        assert APFRenderer().render_batch([]) == ""

    def test_render_batch_single(self) -> None:
        spec = _make_spec()
        renderer = APFRenderer()
        assert renderer.render_batch([spec]) == renderer.render(spec)

    def test_render_batch_multiple(self) -> None:
        specs = [_make_spec(card_index=i) for i in range(1, 3)]
        result = APFRenderer().render_batch(specs)
        assert "<!-- CARD_SEPARATOR -->" in result


# ===================================================================
# APFSentinelValidator
# ===================================================================


class TestAPFSentinelValidator:
    def test_valid_apf(self) -> None:
        validator = APFSentinelValidator()
        apf = APFRenderer().render(_make_spec())
        assert validator.is_valid(apf)
        assert validator.validate(apf) == []

    def test_empty_string(self) -> None:
        validator = APFSentinelValidator()
        missing = validator.validate("")
        assert len(missing) == len(validator.REQUIRED_SENTINELS)
        assert not validator.is_valid("")

    def test_partial_sentinels(self) -> None:
        validator = APFSentinelValidator()
        html = "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->"
        missing = validator.validate(html)
        assert len(missing) > 0


# ===================================================================
# Linter (validate_apf)
# ===================================================================


class TestLinter:
    def test_valid_apf(self) -> None:
        apf = _make_valid_apf()
        result = validate_apf(apf)
        assert isinstance(result, LintResult)
        assert result.is_valid, f"errors: {result.errors}"

    def test_missing_sentinels(self) -> None:
        result = validate_apf("just some text")
        assert not result.is_valid
        assert any("sentinel" in e.lower() or "PROMPT_VERSION" in e for e in result.errors)

    def test_no_card_blocks(self) -> None:
        apf = (
            "<!-- PROMPT_VERSION: apf-v2.1 -->\n"
            "<!-- BEGIN_CARDS -->\n"
            "<!-- END_CARDS -->\n"
            "END_OF_CARDS"
        )
        result = validate_apf(apf)
        assert not result.is_valid
        assert any("No card blocks" in e for e in result.errors)

    def test_duplicate_slugs(self) -> None:
        block = _make_valid_apf()
        # Duplicate the card block inside BEGIN/END
        inner = block.split("<!-- BEGIN_CARDS -->")[1].split("<!-- END_CARDS -->")[0]
        apf = (
            "<!-- PROMPT_VERSION: apf-v2.1 -->\n"
            "<!-- BEGIN_CARDS -->\n"
            f"{inner}\n{inner}\n"
            "<!-- END_CARDS -->\n"
            "END_OF_CARDS"
        )
        result = validate_apf(apf)
        assert any("Duplicate slug" in e for e in result.errors)

    def test_missing_end_of_cards(self) -> None:
        apf = _make_valid_apf().replace("END_OF_CARDS", "")
        result = validate_apf(apf)
        assert any("END_OF_CARDS" in e for e in result.errors)


# ===================================================================
# HTMLTemplateGenerator
# ===================================================================


class TestHTMLTemplateGenerator:
    def test_generate_simple_card(self) -> None:
        gen = HTMLTemplateGenerator()
        result = gen.generate_card_html(
            {
                "card_index": 1,
                "slug": "test-slug",
                "tags": "python testing basics",
                "title": "My Title",
                "key_points": ["Point A", "Point B"],
            },
            template_name="simple",
        )
        assert isinstance(result, GenerationResult)
        assert "My Title" in result.html

    def test_generate_code_block_card(self) -> None:
        gen = HTMLTemplateGenerator()
        result = gen.generate_card_html(
            {
                "card_index": 1,
                "slug": "code-slug",
                "tags": "python code basics",
                "title": "Code Card",
                "code_sample": {"code": "x = 1", "language": "python"},
                "key_points": ["Uses assignment"],
            },
            template_name="code_block",
        )
        assert "language-python" in result.html

    def test_generate_full_apf_html(self) -> None:
        gen = HTMLTemplateGenerator()
        result = gen.generate_full_apf_html(
            [{"title": "Card 1", "key_points": ["p1"]}]
        )
        assert "<!-- PROMPT_VERSION: apf-v2.1 -->" in result
        assert "<!-- BEGIN_CARDS -->" in result
        assert "END_OF_CARDS" in result


# ===================================================================
# Converter (basic fallback -- no mistune dependency required)
# ===================================================================


class TestConverterBasic:
    def test_basic_markdown_bold(self) -> None:
        result = _basic_markdown_to_html("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_basic_markdown_italic(self) -> None:
        result = _basic_markdown_to_html("*italic text*")
        assert "<em>italic text</em>" in result

    def test_basic_markdown_code_block(self) -> None:
        md = "```python\nprint('hi')\n```"
        result = _basic_markdown_to_html(md)
        assert "language-python" in result

    def test_sanitize_html_empty(self) -> None:
        assert sanitize_html("") == ""


# ===================================================================
# Validator
# ===================================================================


class TestValidator:
    def test_validate_card_html_clean(self) -> None:
        html = '<pre><code class="language-python">x = 1</code></pre>'
        errors = validate_card_html(html)
        assert errors == []

    def test_validate_card_html_backtick_fences(self) -> None:
        html = "```python\nx = 1\n```"
        errors = validate_card_html(html)
        assert any("Backtick" in e for e in errors)

    def test_validate_markdown_valid(self) -> None:
        result = validate_markdown("Hello **world**")
        assert isinstance(result, MarkdownValidationResult)
        assert result.is_valid

    def test_validate_markdown_unclosed_fence(self) -> None:
        result = validate_markdown("```python\ncode here")
        assert not result.is_valid
        assert any("Unclosed code fence" in e for e in result.errors)

    def test_validate_markdown_empty(self) -> None:
        result = validate_markdown("")
        assert result.is_valid
