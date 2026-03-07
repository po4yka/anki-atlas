"""Tag normalization, validation, and suggestion functions.

Migrated from claude-code-obsidian-anki/src/utils/tag_taxonomy.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from packages.taxonomy.tags import (
    _TOPIC_PREFIXES,
    ALL_TOPIC_TAGS,
    META_TAG_PREFIXES,
    META_TAGS,
    TAG_MAPPING,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def normalize_tag(tag: str) -> str:
    """Normalize a single tag to canonical form.

    Handles: case folding, prefix standardization, separator normalization.
    Meta tags (difficulty::*, atomic, etc.) are preserved as-is.
    Known tags are mapped via TAG_MAPPING.
    Unknown tags are lowercased with kebab-case.
    """
    tag = tag.strip()
    if not tag:
        return ""

    # Keep meta tags as-is
    if tag in META_TAGS or any(tag.startswith(p) for p in META_TAG_PREFIXES):
        return tag

    # Keep already-prefixed topic tags as-is
    if any(tag.startswith(p) for p in _TOPIC_PREFIXES) or tag == "cognitive_bias":
        return tag

    # Map known tags
    if tag in TAG_MAPPING:
        return TAG_MAPPING[tag]

    # Unknown tag: lowercase, kebab-case
    normalized = tag.lower().replace("_", "-").replace("::", "-").replace("/", "-")
    # Collapse multiple hyphens
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")


def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize a list of tags: normalize, deduplicate, sort.

    Returns:
        Sorted list of normalized, deduplicated tags (empty strings removed).
    """
    seen: set[str] = set()
    for tag in tags:
        result = normalize_tag(tag)
        if result:
            seen.add(result)
    return sorted(seen)


def validate_tag(tag: str) -> list[str]:
    """Validate a tag and return a list of issues (empty if valid).

    Checks:
    - Not empty or whitespace-only
    - Uses ``::`` for domain prefix (not ``_`` or ``/``)
    - Kebab-case within tag parts (no underscores as word separators)
    - Max 2 hierarchy levels (prefix::topic)
    - Lowercase (except known code identifiers)
    """
    issues: list[str] = []

    if not tag or not tag.strip():
        issues.append("Tag is empty or whitespace-only")
        return issues

    tag = tag.strip()

    # Check for underscore as prefix separator (anti-pattern)
    for prefix in ("kotlin_", "android_", "cs_", "bias_"):
        if tag.startswith(prefix) and tag != "cognitive_bias" and tag not in ALL_TOPIC_TAGS:
            issues.append(
                f"Use '::' for domain prefix, not '_': "
                f"'{tag}' -> '{prefix[:-1]}::{tag[len(prefix) :]}'"
            )
            break

    # Check for slash as hierarchy separator
    if "/" in tag:
        issues.append(f"Use '::' for hierarchy, not '/': '{tag}'")

    # Check hierarchy depth
    parts = tag.split("::")
    if len(parts) > 2:
        issues.append(f"Tag too deep (max 2 levels): '{tag}' has {len(parts)} levels")

    # Check for uppercase (except known code identifiers and meta tags)
    if "::" in tag:
        prefix_part = parts[0]
        if prefix_part != prefix_part.lower():
            issues.append(f"Prefix should be lowercase: '{prefix_part}'")
        if len(parts) > 1:
            topic_part = parts[1]
            # Allow code identifiers (e.g., ArrayList, WorkManager)
            if topic_part != topic_part.lower() and not topic_part[0].isupper():
                issues.append(f"Topic should be lowercase: '{topic_part}'")

    # Check for underscores as word separators (within tag parts, not prefix sep)
    for part in parts:
        if "_" in part and tag not in ALL_TOPIC_TAGS and tag != "cognitive_bias":
            issues.append(f"Use '-' between words, not '_': '{part}'")
            break

    # Check for duplicate separators
    if "::" in tag and "::::" in tag:
        issues.append("Duplicate '::' separator found")
    if "--" in tag:
        issues.append("Duplicate '-' separator found")

    return issues


def suggest_tag(input_tag: str) -> list[str]:
    """Suggest close matches for a tag from the known taxonomy.

    Returns up to 5 suggestions sorted by relevance.
    """
    if not input_tag or not input_tag.strip():
        return []

    input_tag = input_tag.strip().lower()

    # Build candidates from TAG_MAPPING keys + values + ALL_TOPIC_TAGS
    candidates = set(TAG_MAPPING.keys()) | set(TAG_MAPPING.values()) | ALL_TOPIC_TAGS
    return _find_close_matches(input_tag, candidates, max_results=5)


def is_meta_tag(tag: str) -> bool:
    """Check if a tag is a meta tag (difficulty::*, lang::*, atomic, etc.)."""
    return tag in META_TAGS or any(tag.startswith(p) for p in META_TAG_PREFIXES)


def is_topic_tag(tag: str) -> bool:
    """Check if a tag is a recognized topic tag."""
    return (
        tag in ALL_TOPIC_TAGS
        or any(tag.startswith(p) for p in _TOPIC_PREFIXES)
        or tag == "cognitive_bias"
    )


def _find_close_matches(
    tag: str,
    candidates: Iterable[str],
    max_results: int = 5,
    max_distance: int = 2,
) -> list[str]:
    """Find close matches for a tag using simple edit distance approximation."""
    tag_lower = tag.lower()
    scored: list[tuple[int, str]] = []

    for candidate in candidates:
        candidate_lower = candidate.lower()

        # Exact match
        if candidate_lower == tag_lower:
            continue

        # Prefix match gets score 0 (best)
        if candidate_lower.startswith(tag_lower) or tag_lower.startswith(candidate_lower):
            scored.append((0, candidate))
            continue

        # Quick length check
        if abs(len(tag) - len(candidate)) > max_distance:
            continue

        # Simple character difference count
        differences = sum(1 for a, b in zip(tag_lower, candidate_lower, strict=False) if a != b)
        differences += abs(len(tag) - len(candidate))

        if differences <= max_distance:
            scored.append((differences, candidate))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [s[1] for s in scored[:max_results]]


# Re-export VALID_PREFIXES for convenience (used in validate_tag docs)
__all__ = [
    "is_meta_tag",
    "is_topic_tag",
    "normalize_tag",
    "normalize_tags",
    "suggest_tag",
    "validate_tag",
]
