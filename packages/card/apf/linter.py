"""APF format validation and linting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

from packages.common.logging import get_logger

logger = get_logger(module=__name__)

# Constants
MAX_LINE_WIDTH: Final[int] = 88
MIN_TAGS: Final[int] = 3
MAX_TAGS: Final[int] = 6

REQUIRED_SENTINELS: Final[tuple[str, ...]] = (
    r"^<!-- PROMPT_VERSION: apf-v2\.1 -->$",
    r"^<!-- BEGIN_CARDS -->$",
    r"^<!-- END_CARDS -->$",
)

FIELD_HEADERS_ORDER: Final[tuple[str, ...]] = (
    "<!-- Title -->",
    "<!-- Subtitle (optional) -->",
    "<!-- Syntax (inline) (optional) -->",
    "<!-- Sample (caption) (optional) -->",
    "<!-- Sample (code block or image) (optional for Missing) -->",
    "<!-- Key point (code block / image) -->",
    "<!-- Key point notes -->",
    "<!-- Other notes (optional) -->",
    "<!-- Markdown (optional) -->",
)

ALLOWED_LANGUAGES: Final[frozenset[str]] = frozenset(
    {
        "kotlin",
        "java",
        "python",
        "javascript",
        "typescript",
        "swift",
        "objective_c",
        "c",
        "cpp",
        "rust",
        "go",
        "dart",
        "ruby",
        "php",
        "csharp",
        "sql",
        "yaml",
        "json",
        "bash",
        "powershell",
        "docker",
        "kubernetes",
        "terraform",
        "ansible",
        "gradle",
        "maven",
        "git",
        "regex",
    }
)

ALLOWED_FIRST_TAGS: Final[frozenset[str]] = ALLOWED_LANGUAGES | frozenset(
    {
        "android",
        "ios",
        "web",
        "mobile",
        "backend",
        "frontend",
        "fullstack",
        "devops",
        "cloud",
        "database",
        "security",
        "testing",
        "architecture",
        "design",
        "ux",
        "ui",
        "api",
        "microservices",
        "serverless",
    }
)


@dataclass(frozen=True, slots=True)
class LintResult:
    """Result of APF linting/validation."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return True when there are no errors."""
        return len(self.errors) == 0


def validate_apf(apf_html: str, slug: str | None = None) -> LintResult:
    """Validate APF card format against specification."""
    errors: list[str] = []
    warnings: list[str] = []
    lines = apf_html.split("\n")

    _check_sentinels(lines, errors)

    if lines and lines[-1].strip() != "END_OF_CARDS":
        errors.append("Missing final 'END_OF_CARDS' line")

    card_blocks = _extract_card_blocks(apf_html)

    if not card_blocks:
        errors.append("No card blocks found")
        return LintResult(errors=tuple(errors), warnings=tuple(warnings))

    for idx, block in enumerate(card_blocks, 1):
        _validate_card_block(block, idx, slug, errors, warnings)

    _check_duplicate_slugs(card_blocks, errors)

    logger.debug(
        "apf_validation_completed",
        slug=slug,
        errors=len(errors),
        warnings=len(warnings),
    )

    return LintResult(errors=tuple(errors), warnings=tuple(warnings))


def _check_sentinels(lines: list[str], errors: list[str]) -> None:
    """Check required sentinel lines."""
    content = "\n".join(lines)

    if not re.search(REQUIRED_SENTINELS[0], content, re.MULTILINE):
        errors.append("Missing '<!-- PROMPT_VERSION: apf-v2.1 -->' sentinel")

    if not re.search(REQUIRED_SENTINELS[1], content, re.MULTILINE):
        errors.append("Missing '<!-- BEGIN_CARDS -->' sentinel")

    if not re.search(REQUIRED_SENTINELS[2], content, re.MULTILINE):
        errors.append("Missing '<!-- END_CARDS -->' sentinel")


def _extract_card_blocks(apf_html: str) -> list[str]:
    """Extract individual card blocks from APF HTML."""
    match = re.search(r"<!-- BEGIN_CARDS -->(.*?)<!-- END_CARDS -->", apf_html, re.DOTALL)
    if not match:
        return []

    cards_content = match.group(1)
    card_pattern = r"(<!-- Card \d+.*?-->.*?)(?=<!-- Card \d+|$)"
    return re.findall(card_pattern, cards_content, re.DOTALL)


def _validate_card_block(
    block: str,
    card_num: int,
    expected_slug: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single card block."""
    lines = block.strip().split("\n")

    if not lines:
        errors.append(f"Card {card_num}: Empty card block")
        return

    header_line = lines[0].strip()

    header_errors: list[str] = []
    _validate_header_format_strict(header_line, card_num, header_errors)
    if header_errors:
        errors.extend(header_errors)
        return

    header_pattern = (
        r"<!-- Card (\d+) \| slug: ([a-z0-9-]+) \| "
        r"CardType: (Simple|Missing|Draw) \| Tags: (.+?) -->"
    )
    header_match = re.match(header_pattern, header_line)

    if not header_match:
        actual_header = header_line[:200]
        logger.warning(
            "invalid_card_header",
            card_num=card_num,
            actual=actual_header,
        )
        errors.append(
            f"Card {card_num}: Invalid card header format. Found: '{actual_header[:100]}...'"
        )
        return

    _card_idx, slug_val, card_type, tags_str = header_match.groups()

    if expected_slug and slug_val != expected_slug:
        warnings.append(
            f"Card {card_num}: Slug mismatch (expected {expected_slug}, got {slug_val})"
        )

    tags = tags_str.strip().split()
    _validate_tags(tags, card_num, errors, warnings)

    if "<!-- manifest:" not in block:
        errors.append(f"Card {card_num}: Missing manifest")
    else:
        _validate_manifest(
            block,
            slug_val,
            card_num,
            errors,
            warnings,
            expected_tags=tags,
            expected_type=card_type,
        )

    _check_field_headers(block, card_num, errors)
    _validate_key_point_notes(block, card_num, warnings)

    if card_type == "Missing":
        _validate_cloze_density(block, card_num, errors, warnings)

    for line_num, line in enumerate(lines, 1):
        if "data:image/svg+xml" in line:
            continue
        if len(line) > MAX_LINE_WIDTH:
            warnings.append(
                f"Card {card_num}, line {line_num}: "
                f"Line exceeds {MAX_LINE_WIDTH} characters ({len(line)})"
            )


def _validate_header_format_strict(header_line: str, card_num: int, errors: list[str]) -> None:
    """Perform strict validation of card header format."""
    if not header_line.startswith("<!--"):
        errors.append(f"Card {card_num}: Header must start with '<!--'")
        return

    if not header_line.endswith("-->"):
        errors.append(f"Card {card_num}: Header must end with '-->'")
        return

    if "type:" in header_line.lower() and "CardType:" not in header_line:
        errors.append(f"Card {card_num}: Use 'CardType:' not 'type:' (case-sensitive)")
        return

    if "cardtype:" in header_line.lower() and "CardType:" not in header_line:
        errors.append(f"Card {card_num}: Use 'CardType:' with capital C and T")
        return

    if " |" not in header_line or "| " not in header_line:
        errors.append(f"Card {card_num}: Header must have spaces around pipe characters: ' | '")
        return

    if "Tags:" in header_line:
        tags_part = header_line.split("Tags:")[1].split("-->")[0]
        if "," in tags_part:
            errors.append(f"Card {card_num}: Tags must be space-separated, not comma-separated")


def _validate_tags(
    tags: list[str],
    card_num: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate tag count and format."""
    if not (MIN_TAGS <= len(tags) <= MAX_TAGS):
        errors.append(f"Card {card_num}: Must have {MIN_TAGS}-{MAX_TAGS} tags, found {len(tags)}")

    normalized_tags = []
    for tag in tags:
        normalized_tag = tag.replace("-", "_")
        normalized_tags.append(normalized_tag)

        if re.search(r"\s", tag):
            errors.append(f"Card {card_num}: Tag '{tag}' must not contain whitespace")
            continue

        if not re.match(r"^\w+$", normalized_tag):
            errors.append(
                f"Card {card_num}: Tag '{tag}' contains invalid characters "
                "(use alphanumeric, '_', or '-')"
            )
            continue

        if not tag.islower():
            warnings.append(f"Card {card_num}: Tag '{tag}' should be lowercase (convention)")

    if tags and tags[0] not in ALLOWED_FIRST_TAGS:
        warnings.append(
            f"Card {card_num}: First tag should be a language/tool/platform, got '{tags[0]}'"
        )

    non_lang_tags = [t for t in normalized_tags if t not in ALLOWED_LANGUAGES]
    if not non_lang_tags:
        errors.append(f"Card {card_num}: Must have at least one non-language tag")


def _validate_manifest(
    block: str,
    expected_slug: str,
    card_num: int,
    errors: list[str],
    warnings: list[str],
    *,
    expected_tags: list[str] | None = None,
    expected_type: str | None = None,
) -> None:
    """Validate manifest JSON."""
    match = re.search(r"<!-- manifest: ({.*?}) -->", block)
    if not match:
        return

    try:
        manifest = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        errors.append(f"Card {card_num}: Invalid manifest JSON: {e}")
        return

    required_fields = ("slug", "lang", "type", "tags")
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        errors.append(f"Card {card_num}: Manifest missing fields: {missing}")

    if manifest.get("slug") != expected_slug:
        errors.append(
            f"Card {card_num}: Manifest slug mismatch "
            f"(header: {expected_slug}, manifest: {manifest.get('slug')})"
        )

    if expected_tags:
        manifest_tags = set(manifest.get("tags", []))
        header_tags = set(expected_tags)
        if manifest_tags != header_tags:
            warnings.append(f"Card {card_num}: Manifest tags do not match header tags")

    if expected_type:
        manifest_type = manifest.get("type", "")
        if expected_type not in manifest_type:
            warnings.append(
                f"Card {card_num}: Manifest type '{manifest_type}' "
                f"mismatch with header '{expected_type}'"
            )


def _validate_key_point_notes(block: str, card_num: int, warnings: list[str]) -> None:
    """Validate content of Key point notes."""
    match = re.search(r"<!-- Key point notes -->\s*(.*?)(?=<!--|\Z)", block, re.DOTALL)
    if not match:
        return

    content = match.group(1).strip()
    if not content:
        return

    has_html_list = "<li>" in content
    has_md_list = bool(re.search(r"^\s*[-*]\s+", content, re.MULTILINE))

    if not has_html_list and not has_md_list:
        warnings.append(f"Card {card_num}: 'Key point notes' should contain a list")

    if has_html_list:
        bullet_count = content.count("<li>")
        if bullet_count < 3:
            warnings.append(
                f"Card {card_num}: 'Key point notes' has few bullets ({bullet_count}), aim for 5-7"
            )


def _check_field_headers(block: str, card_num: int, errors: list[str]) -> None:
    """Check that required field headers are present and non-empty."""
    required = (
        "<!-- Title -->",
        "<!-- Key point (code block / image) -->",
        "<!-- Key point notes -->",
    )

    for header in required:
        if header not in block:
            if (
                header == "<!-- Key point (code block / image) -->"
                and "<!-- Key point -->" in block
            ):
                continue
            errors.append(f"Card {card_num}: Missing required field header '{header}'")
            continue

        escaped_header = re.escape(header)
        pattern = rf"{escaped_header}\s*(.*?)(?=<!--|\Z)"
        match = re.search(pattern, block, re.DOTALL)

        if match:
            content = match.group(1).strip()
            if not content and (header == "<!-- Title -->" or "Key point" in header):
                errors.append(f"Card {card_num}: Field '{header}' is empty")


def _validate_cloze_density(
    block: str, card_num: int, errors: list[str], warnings: list[str]
) -> None:
    """Validate cloze numbering is dense (1..N)."""
    cloze_matches = re.findall(r"\{\{c(\d+)::", block)

    if not cloze_matches:
        warnings.append(f"Card {card_num}: Missing card has no cloze deletions")
        return

    indices = sorted({int(m) for m in cloze_matches})
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        errors.append(
            f"Card {card_num}: Cloze indices not dense (expected {expected}, got {indices})"
        )


def _check_duplicate_slugs(card_blocks: list[str], errors: list[str]) -> None:
    """Check for duplicate slugs across blocks."""
    seen: set[str] = set()
    for block in card_blocks:
        match = re.search(r"slug: ([a-z0-9-]+)", block)
        if match:
            slug_val = match.group(1)
            if slug_val in seen:
                errors.append(f"Duplicate slug found: {slug_val}")
            seen.add(slug_val)
