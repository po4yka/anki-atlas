"""Built-in validators for card content, format, HTML, and tags."""

from __future__ import annotations

import re
from typing import Any

from packages.validation.pipeline import (
    Severity,
    ValidationIssue,
    ValidationResult,
)

_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})

_FORBIDDEN_TAGS = frozenset({"script", "style", "iframe", "object", "applet"})

_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?>")


class ContentValidator:
    """Check card content quality: empty fields, length, code blocks."""

    MIN_LENGTH: int = 10
    MAX_LENGTH: int = 5000

    def validate(
        self,
        *,
        front: str,
        back: str,
        tags: tuple[str, ...] = (),  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if not front.strip():
            issues.append(
                ValidationIssue(Severity.ERROR, "Front side is empty", "front")
            )
        elif len(front.strip()) < self.MIN_LENGTH:
            issues.append(
                ValidationIssue(Severity.WARNING, "Front side is very short", "front")
            )

        if not back.strip():
            issues.append(
                ValidationIssue(Severity.ERROR, "Back side is empty", "back")
            )
        elif len(back.strip()) < self.MIN_LENGTH:
            issues.append(
                ValidationIssue(Severity.WARNING, "Back side is very short", "back")
            )

        if len(front) > self.MAX_LENGTH:
            issues.append(
                ValidationIssue(Severity.WARNING, "Front side exceeds 5000 chars", "front")
            )
        if len(back) > self.MAX_LENGTH:
            issues.append(
                ValidationIssue(Severity.WARNING, "Back side exceeds 5000 chars", "back")
            )

        for label, text in (("front", front), ("back", back)):
            if text.count("```") % 2 != 0:
                issues.append(
                    ValidationIssue(Severity.ERROR, "Unmatched code fence", label)
                )

        return ValidationResult(issues=tuple(issues))


class FormatValidator:
    """Check APF format compliance: whitespace, blank lines."""

    def validate(
        self,
        *,
        front: str,
        back: str,
        tags: tuple[str, ...] = (),  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        for label, text in (("front", front), ("back", back)):
            for i, line in enumerate(text.splitlines(), 1):
                if line != line.rstrip():
                    issues.append(
                        ValidationIssue(
                            Severity.WARNING,
                            f"Trailing whitespace on line {i}",
                            label,
                        )
                    )

            if "\n\n\n" in text:
                issues.append(
                    ValidationIssue(
                        Severity.WARNING, "Consecutive blank lines", label
                    )
                )

        return ValidationResult(issues=tuple(issues))


class HTMLValidator:
    """Validate HTML in card fields: balanced tags, forbidden elements."""

    def validate(
        self,
        *,
        front: str,
        back: str,
        tags: tuple[str, ...] = (),  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        for label, text in (("front", front), ("back", back)):
            issues.extend(self._check_html(text, label))

        return ValidationResult(issues=tuple(issues))

    @staticmethod
    def _check_html(text: str, location: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        stack: list[str] = []

        for match in _TAG_RE.finditer(text):
            is_closing = match.group(1) == "/"
            tag_name = match.group(2).lower()

            if tag_name in _FORBIDDEN_TAGS:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        f"Forbidden HTML tag: <{tag_name}>",
                        location,
                    )
                )
                continue

            if tag_name in _VOID_ELEMENTS:
                continue

            if is_closing:
                if stack and stack[-1] == tag_name:
                    stack.pop()
                else:
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            f"Unexpected closing tag: </{tag_name}>",
                            location,
                        )
                    )
            else:
                stack.append(tag_name)

        for unclosed in stack:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Unclosed HTML tag: <{unclosed}>",
                    location,
                )
            )

        return issues


class TagValidator:
    """Validate tags against conventions."""

    MAX_TAGS: int = 20
    _INVALID_CHARS_RE = re.compile(r"[^a-zA-Z0-9_:/-]")

    def validate(
        self,
        *,
        front: str,  # noqa: ARG002
        back: str,  # noqa: ARG002
        tags: tuple[str, ...] = (),
        **kwargs: Any,  # noqa: ARG002
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        for tag in tags:
            if not tag.strip():
                issues.append(
                    ValidationIssue(Severity.ERROR, "Empty tag", "tags")
                )
            elif self._INVALID_CHARS_RE.search(tag):
                issues.append(
                    ValidationIssue(
                        Severity.WARNING,
                        f"Tag contains invalid characters: {tag!r}",
                        "tags",
                    )
                )

        if len(tags) > self.MAX_TAGS:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"Too many tags ({len(tags)} > {self.MAX_TAGS})",
                    "tags",
                )
            )

        unique = set(tags)
        if len(unique) < len(tags):
            issues.append(
                ValidationIssue(Severity.WARNING, "Duplicate tags found", "tags")
            )

        return ValidationResult(issues=tuple(issues))
