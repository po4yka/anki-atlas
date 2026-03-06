"""Prompt template rendering engine.

Provides PromptTemplate for parsing, substituting, and validating
YAML-frontmatter prompt templates.

Adapted from claude-code-obsidian-anki template_parser.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from packages.common.exceptions import ConfigurationError


def _load_yaml() -> Any:
    """Lazy-import PyYAML."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "PyYAML is required for template parsing: pip install pyyaml"
        ) from exc
    return yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Parsed prompt template with optional YAML frontmatter metadata."""

    deck: str | None = None
    note_type: str | None = None
    field_map: dict[str, str] | None = None
    quality_check: dict[str, Any] | None = None
    prompt_body: str = ""

    def substitute(self, **kwargs: Any) -> str:
        """Replace ``{variable}`` placeholders in *prompt_body*."""
        result = self.prompt_body
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, str(value) if value is not None else "")
        return result

    def get_required_variables(self) -> tuple[str, ...]:
        """Return unique placeholder names found in *prompt_body*."""
        return tuple(sorted(set(re.findall(r"\{(\w+)\}", self.prompt_body))))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_template_string(content: str) -> PromptTemplate:
    """Parse a template string with optional YAML frontmatter.

    Raises:
        ConfigurationError: If YAML frontmatter is malformed.
    """
    yaml = _load_yaml()
    match = _FRONTMATTER_RE.match(content)

    if match:
        yaml_content = match.group(1)
        prompt_body = match.group(2).strip()
        try:
            metadata: dict[str, Any] = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML frontmatter: {exc}") from exc

        return PromptTemplate(
            deck=metadata.get("deck"),
            note_type=metadata.get("note_type"),
            field_map=metadata.get("field_map"),
            quality_check=metadata.get("quality_check"),
            prompt_body=prompt_body,
        )

    return PromptTemplate(prompt_body=content.strip())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_template(template: PromptTemplate) -> tuple[str, ...]:
    """Return validation error messages (empty tuple if valid)."""
    errors: list[str] = []

    if not template.prompt_body:
        errors.append("Prompt body is empty")
    elif len(template.prompt_body) < 50:
        errors.append("Prompt body is too short (< 50 characters)")

    if template.field_map is not None:
        if not isinstance(template.field_map, dict):
            errors.append("field_map must be a dictionary")
        elif not template.field_map:
            errors.append("field_map is empty")

    open_count = template.prompt_body.count("{")
    close_count = template.prompt_body.count("}")
    if open_count != close_count:
        errors.append(
            f"Unbalanced placeholders: {open_count} opening, {close_count} closing braces"
        )

    return tuple(errors)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def create_simple_template(
    prompt_body: str,
    deck: str | None = None,
) -> PromptTemplate:
    """Create a template with just a prompt body and optional deck."""
    return PromptTemplate(deck=deck, prompt_body=prompt_body)


def merge_templates(base: PromptTemplate, override: PromptTemplate) -> PromptTemplate:
    """Merge two templates; *override* values take precedence."""
    merged_field_map: dict[str, str] | None = None
    if base.field_map or override.field_map:
        merged_field_map = {}
        if base.field_map:
            merged_field_map.update(base.field_map)
        if override.field_map:
            merged_field_map.update(override.field_map)

    merged_quality_check: dict[str, Any] | None = None
    if base.quality_check or override.quality_check:
        merged_quality_check = {}
        if base.quality_check:
            merged_quality_check.update(base.quality_check)
        if override.quality_check:
            merged_quality_check.update(override.quality_check)

    return PromptTemplate(
        deck=override.deck or base.deck,
        note_type=override.note_type or base.note_type,
        field_map=merged_field_map,
        quality_check=merged_quality_check,
        prompt_body=override.prompt_body or base.prompt_body,
    )
