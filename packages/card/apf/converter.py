"""Convert APF Markdown content to HTML for Anki.

Uses mistune for fast Markdown parsing, Pygments for syntax highlighting,
and nh3 for HTML sanitization.  All three are lazy-imported so that
importing this module does not require them at package import time.
"""

from __future__ import annotations

import re
from html import escape
from typing import TYPE_CHECKING, Any, Final

from packages.common.logging import get_logger

if TYPE_CHECKING:
    import mistune

logger = get_logger(module=__name__)

# Allowed HTML tags for Anki cards (used by nh3 sanitizer)
ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "p",
        "br",
        "strong",
        "b",
        "em",
        "i",
        "u",
        "s",
        "strike",
        "code",
        "pre",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "a",
        "img",
        "div",
        "span",
        "sup",
        "sub",
        "hr",
        "figure",
        "figcaption",
    }
)

_GLOBAL_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "class",
        "id",
        "style",
    }
)

_TAG_SPECIFIC_ATTRIBUTES: Final[dict[str, set[str]]] = {
    "a": {"href", "title", "target"},
    "img": {"src", "alt", "title", "width", "height"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
}


def _build_allowed_attributes() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for tag in ALLOWED_TAGS:
        tag_attrs = _TAG_SPECIFIC_ATTRIBUTES.get(tag, set())
        result[tag] = set(_GLOBAL_ATTRIBUTES) | tag_attrs
    return result


ALLOWED_ATTRIBUTES: Final[dict[str, set[str]]] = _build_allowed_attributes()

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_MISTUNE_CONVERTER: dict[str, mistune.Markdown] = {}


def _get_mistune() -> Any:
    """Lazy-import mistune."""
    import mistune as _mistune

    return _mistune


def _get_nh3() -> Any:
    """Lazy-import nh3."""
    import nh3 as _nh3

    return _nh3


def _get_pygments() -> tuple[Any, Any, Any, Any]:
    """Lazy-import pygments components."""
    from pygments import highlight as _highlight
    from pygments.formatters import HtmlFormatter as _HtmlFormatter
    from pygments.lexers import get_lexer_by_name as _get_lexer
    from pygments.lexers import guess_lexer as _guess

    return _highlight, _HtmlFormatter, _get_lexer, _guess


# ---------------------------------------------------------------------------
# Mistune renderer
# ---------------------------------------------------------------------------


def _create_anki_renderer() -> Any:
    """Create an AnkiHighlightRenderer (mistune.HTMLRenderer subclass)."""
    mistune_mod = _get_mistune()
    _highlight_fn, HtmlFormatter, get_lexer_by_name, guess_lexer = _get_pygments()

    class AnkiHighlightRenderer(mistune_mod.HTMLRenderer):  # type: ignore[name-defined,misc]
        """Custom mistune renderer with Pygments syntax highlighting."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._formatter = HtmlFormatter(
                cssclass="codehilite",
                linenos=False,
                nowrap=False,
            )

        def block_code(self, code: str, info: str | None = None) -> str:
            from pygments.util import ClassNotFound

            lang = info.split()[0] if info else None
            try:
                lexer = get_lexer_by_name(lang, stripall=True) if lang else guess_lexer(code)
            except ClassNotFound:
                lang_class = f"language-{lang}" if lang else "language-text"
                escaped_code = escape(code.strip())
                return f'<pre><code class="{lang_class}">{escaped_code}</code></pre>\n'
            result: str = _highlight_fn(code, lexer, self._formatter)
            return result

        def codespan(self, text: str) -> str:
            escaped = escape(text)
            return f'<code class="language-text">{escaped}</code>'

        def paragraph(self, text: str) -> str:
            text = _restore_cloze_syntax(text)
            return f"<p>{text}</p>\n"

    return AnkiHighlightRenderer


def _create_mistune_converter() -> mistune.Markdown:
    """Create a configured mistune Markdown converter."""
    mistune_mod = _get_mistune()
    renderer_cls = _create_anki_renderer()
    renderer = renderer_cls()
    return mistune_mod.create_markdown(
        renderer=renderer,
        plugins=["strikethrough", "table", "task_lists", "footnotes"],
    )


def _get_mistune_converter() -> mistune.Markdown:
    """Get or create cached mistune converter."""
    if "instance" not in _MISTUNE_CONVERTER:
        _MISTUNE_CONVERTER["instance"] = _create_mistune_converter()
    return _MISTUNE_CONVERTER["instance"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _restore_cloze_syntax(text: str) -> str:
    """Restore cloze syntax that may have been escaped during conversion."""
    patterns = (
        (r"\{\{c(\d+)::(.*?)\}\}", r"{{c\1::\2}}"),
        (r"&#123;&#123;c(\d+)::(.*?)&#125;&#125;", r"{{c\1::\2}}"),
        (r"&lbrace;&lbrace;c(\d+)::(.*?)&rbrace;&rbrace;", r"{{c\1::\2}}"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    return text


def _ensure_code_language_classes(html_str: str) -> str:
    """Ensure all code blocks have language-* classes for Anki styling."""

    def add_language_class(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "language-" in tag or 'class="' in tag:
            return tag
        return '<code class="language-text">'

    return re.sub(r"<code(?:\s[^>]*)?>", add_language_class, html_str)


def _is_already_html(content: str) -> bool:
    """Check if content is already HTML (not Markdown)."""
    _html_tags = (
        "p|pre|code|strong|em|b|i|ul|ol|li|table|tr|td|th|div|span|br|hr|blockquote|h[1-6]|a|img"
    )
    html_tag_pattern = re.compile(
        rf"<(?:{_html_tags})(?:\s[^>]*)?/?>",
        re.IGNORECASE,
    )
    return bool(html_tag_pattern.search(content))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_markdown_to_html(md_content: str, *, sanitize: bool = True) -> str:
    """Convert Markdown content to HTML using mistune."""
    if not md_content or not md_content.strip():
        return md_content

    try:
        converter = _get_mistune_converter()
        result = converter(md_content)
        html_str: str = result if isinstance(result, str) else str(result)
    except Exception as e:
        logger.warning("mistune_conversion_failed", error=str(e))
        html_str = _basic_markdown_to_html(md_content)

    html_str = _ensure_code_language_classes(html_str)
    html_str = _restore_cloze_syntax(html_str)

    if sanitize:
        html_str = sanitize_html(html_str)

    return html_str


def _basic_markdown_to_html(md_content: str) -> str:
    """Basic Markdown to HTML conversion fallback (no external deps)."""
    html_str = md_content

    html_str = re.sub(
        r"```(\w*)\n(.*?)\n```",
        lambda m: (
            f'<pre><code class="language-{m.group(1) or "text"}">{escape(m.group(2))}</code></pre>'
        ),
        html_str,
        flags=re.DOTALL,
    )

    html_str = re.sub(
        r"`([^`]+)`",
        r'<code class="language-text">\1</code>',
        html_str,
    )

    html_str = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html_str)

    html_str = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", html_str)

    html_str = re.sub(r"^- (.+)$", r"<li>\1</li>", html_str, flags=re.MULTILINE)
    html_str = re.sub(r"(<li>[^<]*</li>\n?)+", r"<ul>\g<0></ul>", html_str)

    html_str = html_str.replace("\n\n", "</p><p>")
    html_str = f"<p>{html_str}</p>"
    html_str = html_str.replace("<p></p>", "")

    return html_str


def sanitize_html(html_str: str) -> str:
    """Sanitize HTML using nh3.

    Raises:
        RuntimeError: If sanitization fails (never returns unsanitized HTML).
    """
    if not html_str:
        return html_str

    nh3 = _get_nh3()
    sanitized: str = nh3.clean(
        html_str,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        link_rel="noopener noreferrer",
        strip_comments=False,
    )
    return sanitized


def convert_apf_field_to_html(field_content: str) -> str:
    """Convert a single APF field from Markdown to HTML."""
    if not field_content:
        return ""

    if _is_already_html(field_content):
        logger.debug("field_already_html", preview=field_content[:100])
        return sanitize_html(field_content)

    return convert_markdown_to_html(field_content)


def convert_apf_markdown_to_html(apf_markdown: str) -> str:
    """Convert APF Markdown document to APF HTML document.

    Preserves APF structure (sentinels, headers, field markers) while
    converting field content from Markdown to HTML.
    """
    if not apf_markdown:
        return apf_markdown

    if _is_already_html(apf_markdown):
        logger.debug("apf_already_html")
        return apf_markdown

    result_parts: list[str] = []
    current_pos = 0

    field_pattern = re.compile(r"(<!--\s*[^>]+\s*-->)")

    for match in field_pattern.finditer(apf_markdown):
        content_before = apf_markdown[current_pos : match.start()]
        if content_before.strip():
            converted = convert_apf_field_to_html(content_before.strip())
            leading_ws = len(content_before) - len(content_before.lstrip())
            trailing_ws = len(content_before) - len(content_before.rstrip())
            result_parts.append(content_before[:leading_ws])
            result_parts.append(converted)
            if trailing_ws > 0:
                result_parts.append(content_before[-trailing_ws:])
        else:
            result_parts.append(content_before)

        result_parts.append(match.group(0))
        current_pos = match.end()

    remaining = apf_markdown[current_pos:]
    if remaining.strip():
        converted = convert_apf_field_to_html(remaining.strip())
        leading_ws = len(remaining) - len(remaining.lstrip())
        result_parts.append(remaining[:leading_ws])
        result_parts.append(converted)
    else:
        result_parts.append(remaining)

    return "".join(result_parts)


def highlight_code(code: str, language: str | None = None) -> str:
    """Highlight code using Pygments."""
    from pygments.util import ClassNotFound

    try:
        _highlight_fn, HtmlFormatter, get_lexer_by_name, guess_lexer = _get_pygments()

        lexer = get_lexer_by_name(language, stripall=True) if language else guess_lexer(code)
        formatter = HtmlFormatter(
            cssclass="codehilite",
            linenos=False,
            nowrap=False,
        )
        result: str = _highlight_fn(code, lexer, formatter)
        return result
    except ClassNotFound:
        escaped = escape(code)
        lang_class = f"language-{language}" if language else "language-text"
        return f'<pre><code class="{lang_class}">{escaped}</code></pre>'


def get_pygments_css(style: str = "default") -> str:
    """Get CSS for Pygments syntax highlighting."""
    _, HtmlFormatter, _, _ = _get_pygments()
    formatter = HtmlFormatter(style=style, cssclass="codehilite")
    result: str = formatter.get_style_defs()
    return result
