"""Validation pipeline types and orchestration.

Defines the core types (Severity, ValidationIssue, ValidationResult) and the
ValidationPipeline that chains multiple validators in sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class Severity(StrEnum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Single validation issue with severity, message, and optional location."""

    severity: Severity
    message: str
    location: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregated result of running one or more validators."""

    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """True when no ERROR-level issues exist."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return only ERROR-level issues."""
        return tuple(i for i in self.issues if i.severity == Severity.ERROR)

    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return only WARNING-level issues."""
        return tuple(i for i in self.issues if i.severity == Severity.WARNING)

    @staticmethod
    def merge(*results: ValidationResult) -> ValidationResult:
        """Combine multiple results into one."""
        all_issues: list[ValidationIssue] = []
        for r in results:
            all_issues.extend(r.issues)
        return ValidationResult(issues=tuple(all_issues))

    @staticmethod
    def ok() -> ValidationResult:
        """Create an empty (passing) result."""
        return ValidationResult(issues=())


class Validator(Protocol):
    """Protocol that all validators must satisfy."""

    def validate(
        self,
        *,
        front: str,
        back: str,
        tags: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> ValidationResult: ...


class ValidationPipeline:
    """Runs a sequence of validators and aggregates their results."""

    def __init__(self, validators: Sequence[Validator]) -> None:
        self._validators = tuple(validators)

    def run(
        self,
        *,
        front: str,
        back: str,
        tags: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> ValidationResult:
        """Execute all validators and merge results."""
        if not self._validators:
            return ValidationResult.ok()
        results = [
            v.validate(front=front, back=back, tags=tags, **kwargs) for v in self._validators
        ]
        return ValidationResult.merge(*results)
