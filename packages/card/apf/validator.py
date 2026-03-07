"""HTML and Markdown structure validation for APF cards."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarkdownValidationResult:
    """Result of Markdown validation."""

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_card_html(apf_html: str) -> list[str]:
    """Validate HTML structure used in APF cards.

    Returns list of validation error messages (empty if valid).
    BeautifulSoup is lazy-imported to avoid requiring it at package import.
    """
    errors: list[str] = []

    # Regex-based checks (no external deps)
    if "```" in apf_html:
        errors.append("Backtick code fences detected; use <pre><code> blocks.")

    text_without_code = re.sub(r"<pre>.*?</pre>", "", apf_html, flags=re.DOTALL)
    if re.search(r"\*\*[^*]+\*\*", text_without_code):
        errors.append("Markdown **bold** detected; use <strong> HTML tags instead.")
    if re.search(r"(?<!\*)\*[^*\s][^*]*[^*\s]\*(?!\*)", text_without_code):
        errors.append("Markdown *italic* detected; use <em> HTML tags instead.")

    # DOM-based checks (require bs4)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(apf_html, "html.parser")
    except ImportError:
        return errors
    except Exception as exc:
        errors.append(f"HTML parser error: {exc}")
        return errors

    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code is None:
            errors.append("<pre> block without nested <code> element")

    for code in soup.find_all("code"):
        if code.parent is not None and code.parent.name != "pre":
            errors.append("Inline <code> elements are not allowed; wrap in <pre><code>.")

    return errors


# ---------------------------------------------------------------------------
# Markdown validation
# ---------------------------------------------------------------------------


def validate_markdown(content: str) -> MarkdownValidationResult:
    """Validate Markdown structure."""
    errors: list[str] = []
    warnings: list[str] = []

    if not content or not content.strip():
        return MarkdownValidationResult(is_valid=True)

    errors.extend(_validate_code_fences(content))
    errors.extend(_validate_formatting_markers(content))
    warnings.extend(_check_common_issues(content))

    return MarkdownValidationResult(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _validate_code_fences(content: str) -> list[str]:
    """Validate that code fences are properly balanced."""
    errors: list[str] = []
    fence_pattern = r"^```[\w]*\s*$"
    lines = content.split("\n")
    fence_stack: list[int] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(fence_pattern, stripped) or stripped == "```":
            if fence_stack:
                fence_stack.pop()
            else:
                fence_stack.append(i)

    if fence_stack:
        errors.append(
            f"Unclosed code fence starting at line {fence_stack[0]}. "
            "Each ``` must have a matching closing ```."
        )
    return errors


def _validate_formatting_markers(content: str) -> list[str]:
    """Validate that formatting markers are balanced."""
    errors: list[str] = []
    content_without_code = _remove_code_blocks(content)

    backtick_count = content_without_code.count("`")
    if backtick_count % 2 != 0:
        errors.append(
            "Unbalanced inline code markers (`). Each opening backtick needs a closing one."
        )

    bold_marker_count = len(re.findall(r"\*\*", content_without_code))
    if bold_marker_count % 2 != 0:
        errors.append("Unbalanced bold markers (**). Each ** needs a matching closing **.")
    return errors


def _remove_code_blocks(content: str) -> str:
    """Remove code blocks from content for validation purposes."""
    result = re.sub(r"```[\w]*\n.*?```", "", content, flags=re.DOTALL)
    result = re.sub(r"`[^`]+`", "", result)
    return result


def _check_common_issues(content: str) -> list[str]:
    """Check for common Markdown issues."""
    warnings: list[str] = []

    html_tags = ("<pre>", "<code>", "<strong>", "<em>", "<ul>", "<ol>")
    content_lower = content.lower()
    for tag in html_tags:
        if tag in content_lower:
            warnings.append(
                f"HTML tag {tag} found in Markdown content. Consider using Markdown syntax instead."
            )
            break

    lines = content.split("\n")
    in_code_block = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        elif in_code_block and len(line) > 200:
            warnings.append(
                f"Line {i} in code block exceeds 200 characters. "
                "Consider breaking into multiple lines."
            )

    return warnings


def validate_apf_markdown(apf_content: str) -> MarkdownValidationResult:
    """Validate APF document with Markdown content."""
    errors: list[str] = []
    warnings: list[str] = []

    if not apf_content or not apf_content.strip():
        return MarkdownValidationResult(
            is_valid=False,
            errors=("Empty APF content",),
        )

    if "<!-- BEGIN_CARDS -->" not in apf_content:
        errors.append("Missing BEGIN_CARDS sentinel")

    if "<!-- END_CARDS -->" not in apf_content:
        errors.append("Missing END_CARDS sentinel")

    card_header_pattern = r"<!-- Card \d+ \| slug: [a-z0-9-]+ \| CardType: \w+"
    if not re.search(card_header_pattern, apf_content):
        errors.append("Missing or invalid card header")

    if "<!-- Title -->" not in apf_content:
        errors.append("Missing Title section")

    has_key_point = "<!-- Key point" in apf_content
    has_key_notes = "<!-- Key point notes -->" in apf_content
    if not has_key_point and not has_key_notes:
        warnings.append(
            "Missing Key point sections. Cards should have Key point or Key point notes."
        )

    content_pattern = r"<!--[^>]+-->\s*([^<]+?)(?=<!--|$)"
    for match in re.finditer(content_pattern, apf_content, re.DOTALL):
        section_content = match.group(1).strip()
        if section_content:
            md_result = validate_markdown(section_content)
            errors.extend(md_result.errors)
            warnings.extend(md_result.warnings)

    return MarkdownValidationResult(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
