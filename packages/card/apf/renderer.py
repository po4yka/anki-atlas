"""APF Renderer - Convert JSON CardSpec to APF v2.1 HTML.

This module provides deterministic conversion from structured JSON
card specifications to APF HTML format, eliminating the risk of
truncated or malformed output from LLMs.
"""

from __future__ import annotations

import html
import json
from typing import Any


class APFRenderer:
    """Convert JSON CardSpec to APF v2.1 HTML.

    This renderer produces deterministic, well-formed APF output
    from structured card specifications. All required sentinels
    and markers are guaranteed to be present.

    The ``spec`` parameter uses duck typing (any object with the expected
    attributes) so that this package does not depend on the generation
    package where ``CardSpec`` is defined.
    """

    PROMPT_VERSION = "apf-v2.1"

    def render(self, spec: Any) -> str:
        """Render a CardSpec to APF v2.1 HTML."""
        parts = [
            f"<!-- PROMPT_VERSION: {self.PROMPT_VERSION} -->",
            "<!-- BEGIN_CARDS -->",
            "",
            self._render_card_header(spec),
            "",
            self._render_title(spec.front.title),
            "",
            self._render_key_point(spec.front),
            "",
            self._render_key_point_notes(spec.front.key_point_notes),
            "",
            self._render_other_notes(spec.front.other_notes),
            "",
            self._render_extra(spec.front.extra),
            "",
            self._render_manifest(spec),
            "<!-- END_CARDS -->",
            "END_OF_CARDS",
        ]
        return "\n".join(parts)

    def render_batch(self, specs: list[Any]) -> str:
        """Render multiple cards separated by card markers."""
        if not specs:
            return ""
        if len(specs) == 1:
            return self.render(specs[0])

        rendered_cards = [self.render(spec) for spec in specs]
        return "\n\n<!-- CARD_SEPARATOR -->\n\n".join(rendered_cards)

    def _render_card_header(self, spec: Any) -> str:
        """Render the card header comment."""
        tags_str = " ".join(spec.tags) if spec.tags else ""
        return (
            f"<!-- Card {spec.card_index} | slug: {spec.slug} | "
            f"CardType: {spec.card_type} | Tags: {tags_str} -->"
        )

    def _render_title(self, title: str) -> str:
        """Render the title section."""
        return f"<!-- Title -->\n{title}"

    def _render_key_point(self, section: Any) -> str:
        """Render the key point (code block) section."""
        header = "<!-- Key point (code block / image) -->"

        if not section.key_point_code:
            return header

        lang = section.key_point_code_lang or "plaintext"
        escaped_code = html.escape(section.key_point_code)

        return (
            f'{header}\n<pre><code class="language-{lang}">'
            f"{escaped_code}</code></pre>"
        )

    def _render_key_point_notes(self, notes: list[str]) -> str:
        """Render the key point notes section."""
        header = "<!-- Key point notes -->"

        if not notes:
            return f"{header}\n<ul></ul>"

        items = [f"<li>{note}</li>" for note in notes]
        items_str = "\n".join(items)
        return f"{header}\n<ul>\n{items_str}\n</ul>"

    def _render_other_notes(self, other_notes: str) -> str:
        """Render the other notes section."""
        header = "<!-- Other notes -->"

        if not other_notes:
            return header

        return f"{header}\n{other_notes}"

    def _render_extra(self, extra: str) -> str:
        """Render the extra section."""
        header = "<!-- Extra -->"

        if not extra:
            return header

        return f"{header}\n{extra}"

    def _render_manifest(self, spec: Any) -> str:
        """Render the manifest comment."""
        manifest: dict[str, Any] = {
            "slug": spec.slug,
            "slug_base": spec.slug_base or spec.slug.rsplit("-", 2)[0],
            "lang": spec.lang,
            "type": spec.card_type,
            "tags": spec.tags,
            "guid": spec.guid,
        }

        if spec.source_path:
            manifest["source_path"] = spec.source_path
        if spec.source_anchor:
            manifest["source_anchor"] = spec.source_anchor

        manifest_json = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        )
        return f"<!-- manifest: {manifest_json} -->"


class APFSentinelValidator:
    """Validate APF structure has all required sentinels."""

    REQUIRED_SENTINELS = (
        "<!-- PROMPT_VERSION:",
        "<!-- BEGIN_CARDS -->",
        "<!-- Card ",
        "<!-- Title -->",
        "<!-- Key point",
        "<!-- manifest:",
        "<!-- END_CARDS -->",
    )

    def validate(self, apf_html: str) -> list[str]:
        """Return list of missing sentinel markers (empty if all present)."""
        if not apf_html:
            return list(self.REQUIRED_SENTINELS)

        return [s for s in self.REQUIRED_SENTINELS if s not in apf_html]

    def is_valid(self, apf_html: str) -> bool:
        """Return True if all required sentinels are present."""
        return len(self.validate(apf_html)) == 0
