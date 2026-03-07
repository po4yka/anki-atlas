"""Structured HTML generation with templates and validation.

Provides template-based APF HTML generation with built-in validation
to ensure consistent, well-formed HTML output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from typing import Any

import structlog

from packages.card.apf.validator import validate_card_html

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CardTemplate:
    """Template for generating APF card HTML."""

    card_type: str
    sections: dict[str, str]

    def render(self, data: dict[str, Any]) -> str:
        """Render the template with provided data."""
        html_parts: list[str] = []

        for section_name, template in self.sections.items():
            if data.get(section_name):
                rendered = self._render_section(template, data)
                if rendered:
                    html_parts.append(rendered)

        return "\n".join(html_parts)

    def _render_section(self, template: str, data: dict[str, Any]) -> str:
        """Render a single section template."""
        result = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of HTML generation with validation."""

    html: str
    is_valid: bool
    validation_errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class HTMLTemplateGenerator:
    """Structured HTML generation for APF cards with validation."""

    def __init__(self) -> None:
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> dict[str, CardTemplate]:
        """Initialize available card templates."""
        return {
            "simple": CardTemplate(
                card_type="Simple",
                sections={
                    "header": (
                        "<!-- Card {card_index} | slug: {slug} | "
                        "CardType: Simple | Tags: {tags} -->"
                    ),
                    "title": "\n<!-- Title -->\n{title}",
                    "question": "\n<!-- Question -->\n{question}",
                    "answer": "\n<!-- Answer -->\n{answer}",
                    "key_points": "\n<!-- Key point -->\n{key_points}",
                    "notes": "\n<!-- Other notes -->\n{other_notes}",
                    "references": "\n<!-- References -->\n{references}",
                },
            ),
            "code_block": CardTemplate(
                card_type="Simple",
                sections={
                    "header": (
                        "<!-- Card {card_index} | slug: {slug} | "
                        "CardType: Simple | Tags: {tags} -->"
                    ),
                    "title": "\n<!-- Title -->\n{title}",
                    "code_sample": ("\n<!-- Sample (code block) -->\n{code_sample}"),
                    "key_points": "\n<!-- Key point -->\n{key_points}",
                    "notes": "\n<!-- Other notes -->\n{other_notes}",
                },
            ),
            "cloze": CardTemplate(
                card_type="Missing",
                sections={
                    "header": (
                        "<!-- Card {card_index} | slug: {slug} | "
                        "CardType: Missing | Tags: {tags} -->"
                    ),
                    "title": "\n<!-- Title -->\n{title}",
                    "content": "\n<!-- Content -->\n{content}",
                },
            ),
        }

    def generate_card_html(
        self, card_data: dict[str, Any], template_name: str = "simple"
    ) -> GenerationResult:
        """Generate APF HTML for a card using templates."""
        template = self.templates.get(template_name, self.templates["simple"])
        processed_data = self._preprocess_card_data(card_data)
        html_content = template.render(processed_data)

        validation_errors = validate_card_html(html_content)
        warnings: list[str] = []

        if validation_errors:
            html_content, fix_warnings = self._auto_fix_html_issues(html_content, validation_errors)
            warnings.extend(fix_warnings)

            final_errors = validate_card_html(html_content)
            if final_errors:
                validation_errors.extend(final_errors)
                warnings.append("Some HTML validation issues could not be auto-fixed")

        return GenerationResult(
            html=html_content,
            is_valid=len(validation_errors) == 0,
            validation_errors=tuple(validation_errors),
            warnings=tuple(warnings),
        )

    def _preprocess_card_data(self, card_data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess card data for template rendering."""
        processed = card_data.copy()

        processed.setdefault("card_index", 1)
        processed.setdefault("slug", f"card-{processed['card_index']}")
        processed.setdefault("tags", "")
        processed.setdefault("title", "Untitled Card")

        if "code_sample" in processed:
            processed["code_sample"] = self._generate_code_block(processed["code_sample"])

        if "key_points" in processed:
            processed["key_points"] = self._generate_key_points(processed["key_points"])

        for text_field in ("question", "answer", "title", "other_notes", "content"):
            if processed.get(text_field):
                processed[text_field] = self._escape_and_format_text(processed[text_field])

        return processed

    def _generate_code_block(self, code_data: Any) -> str:
        """Generate properly formatted code block HTML."""
        if isinstance(code_data, str):
            return self._create_code_html(code_data, "text")
        if isinstance(code_data, dict):
            code = code_data.get("code", "")
            language = code_data.get("language", "text")
            caption = code_data.get("caption", "")
            return self._create_code_html(code, language, caption)
        return "<pre><code>Invalid code data</code></pre>"

    def _create_code_html(self, code: str, language: str, caption: str = "") -> str:
        """Create properly formatted code HTML."""
        escaped_code = escape(code.strip())
        code_html = f'<pre><code class="language-{language}">{escaped_code}</code></pre>'

        if caption:
            return f"<figure>\n{code_html}\n<figcaption>{escape(caption)}</figcaption>\n</figure>"
        return code_html

    def _generate_key_points(self, key_points: Any) -> str:
        """Generate key points HTML."""
        if isinstance(key_points, str):
            return f"<ul>\n<li>{escape(key_points)}</li>\n</ul>"
        if isinstance(key_points, list):
            points_html = "\n".join(f"<li>{escape(point)}</li>" for point in key_points)
            return f"<ul>\n{points_html}\n</ul>"
        return "<ul><li>Key points data</li></ul>"

    def _escape_and_format_text(self, text: str) -> str:
        """Escape and format text content."""
        if not text:
            return ""

        escaped = escape(text)
        escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.*?)\*", r"<em>\1</em>", escaped)
        escaped = escaped.replace("\n", "<br>")
        return escaped

    def _auto_fix_html_issues(self, html_str: str, errors: list[str]) -> tuple[str, list[str]]:
        """Attempt to auto-fix common HTML validation issues."""
        fixed_html = html_str
        warnings: list[str] = []

        for error in errors:
            if "language- class" in error:
                fixed_html, lang_warnings = self._add_missing_language_classes(fixed_html)
                warnings.extend(lang_warnings)
            elif "wrap in <pre><code>" in error:
                warnings.append("Standalone code wrapping not fully implemented")
            elif "Backtick code fences detected" in error:
                fixed_html = re.sub(
                    r"```[^\n]*\n(.*?)\n```",
                    r"<pre><code>\1</code></pre>",
                    fixed_html,
                    flags=re.DOTALL,
                )
                warnings.append("Converted markdown code fences to HTML")

        return fixed_html, warnings

    def _add_missing_language_classes(self, html_str: str) -> tuple[str, list[str]]:
        """Add default language classes to code elements missing them."""
        warnings: list[str] = []

        def add_class(match: re.Match[str]) -> str:
            code_tag = match.group(0)
            if "class=" not in code_tag:
                warnings.append("Added default language class to code element")
                return code_tag.replace("<code", '<code class="language-text"', 1)
            return code_tag

        pattern = r"<code(?:\s[^>]*)?>"
        fixed_html = re.sub(pattern, add_class, html_str)
        return fixed_html, warnings

    def generate_full_apf_html(
        self,
        cards_data: list[dict[str, Any]],
    ) -> str:
        """Generate complete APF HTML document with all cards."""
        parts: list[str] = [
            "<!-- PROMPT_VERSION: apf-v2.1 -->",
            "<!-- BEGIN_CARDS -->",
            "",
        ]

        for card_data in cards_data:
            result = self.generate_card_html(card_data)
            parts.append(result.html)
            parts.append("")

            if result.warnings:
                logger.warning(
                    "card_generation_warnings",
                    card_index=card_data.get("card_index"),
                    warnings=result.warnings,
                )

        parts.append("<!-- END_CARDS -->")
        parts.append("END_OF_CARDS")

        return "\n".join(parts)
