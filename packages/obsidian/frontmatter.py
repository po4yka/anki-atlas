"""YAML frontmatter parsing and writing for Obsidian notes.

Uses python-frontmatter and ruamel.yaml via lazy imports (optional deps).
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any

import structlog

from packages.common.exceptions import ObsidianParseError

log = structlog.get_logger(__name__)


def _get_frontmatter_lib() -> Any:
    """Lazy import for python-frontmatter."""
    try:
        import frontmatter

        return frontmatter
    except ImportError as e:
        msg = "python-frontmatter is required: pip install anki-atlas[obsidian]"
        raise ObsidianParseError(msg) from e


def _get_ruamel_yaml() -> Any:
    """Lazy import for ruamel.yaml YAML class."""
    try:
        from ruamel.yaml import YAML

        return YAML
    except ImportError as e:
        msg = "ruamel.yaml is required: pip install anki-atlas[obsidian]"
        raise ObsidianParseError(msg) from e


def _preprocess_yaml_frontmatter(content: str) -> str:
    """Fix common YAML syntax errors in frontmatter.

    Fixes:
    - Backticks in YAML values (replaces with plain text)
    - Orphaned list items after inline arrays
    """
    match = re.match(r"^(---\s*\n)(.*?)(\n---\s*\n)", content, re.DOTALL)
    if not match:
        return content

    start = match.group(1)
    body = match.group(2)
    end = match.group(3)
    rest = content[match.end() :]

    # Remove backticks from YAML values
    body = re.sub(r"`([^`]+)`", r"\1", body)

    return start + body + end + rest


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from note content.

    Args:
        content: Full note content including frontmatter delimiters.

    Returns:
        Dictionary of frontmatter key-value pairs. Empty dict if no frontmatter.

    Raises:
        ObsidianParseError: If YAML parsing fails.
    """
    fm = _get_frontmatter_lib()

    try:
        preprocessed = _preprocess_yaml_frontmatter(content)
    except Exception:
        log.debug("yaml_preprocessing_failed")
        preprocessed = content

    try:
        post = fm.loads(preprocessed)
    except Exception as e:
        msg = f"Invalid YAML frontmatter: {e}"
        raise ObsidianParseError(msg) from e

    return dict(post.metadata) if post.metadata else {}


def write_frontmatter(data: dict[str, Any], content: str) -> str:
    """Write or replace YAML frontmatter in note content.

    Args:
        data: Frontmatter key-value pairs to write.
        content: Full note content (may or may not have existing frontmatter).

    Returns:
        Content with updated frontmatter.

    Raises:
        ObsidianParseError: If YAML serialization fails.
    """
    YAML = _get_ruamel_yaml()
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.width = 4096

    try:
        output = StringIO()
        yaml.dump(data, output)
        frontmatter_text = output.getvalue()
    except Exception as e:
        msg = f"Failed to serialize frontmatter: {e}"
        raise ObsidianParseError(msg) from e

    # Strip existing frontmatter if present
    match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    body = content[match.end() :] if match else content

    return f"---\n{frontmatter_text}---\n{body}"
