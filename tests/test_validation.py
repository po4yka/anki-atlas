"""Tests for packages.validation: pipeline, validators, and quality scoring."""

from __future__ import annotations

import pytest

from packages.validation.pipeline import (
    Severity,
    ValidationIssue,
    ValidationPipeline,
    ValidationResult,
)
from packages.validation.quality import QualityScore, assess_quality
from packages.validation.validators import (
    ContentValidator,
    FormatValidator,
    HTMLValidator,
    TagValidator,
)

# ---------------------------------------------------------------------------
# ValidationIssue / ValidationResult
# ---------------------------------------------------------------------------


class TestValidationIssue:
    def test_create(self) -> None:
        issue = ValidationIssue(Severity.ERROR, "bad", "front")
        assert issue.severity == Severity.ERROR
        assert issue.message == "bad"
        assert issue.location == "front"

    def test_default_location(self) -> None:
        issue = ValidationIssue(Severity.WARNING, "warn")
        assert issue.location == ""

    def test_frozen(self) -> None:
        issue = ValidationIssue(Severity.INFO, "info")
        with pytest.raises(AttributeError):
            issue.message = "changed"  # type: ignore[misc]


class TestValidationResult:
    def test_ok(self) -> None:
        r = ValidationResult.ok()
        assert r.is_valid is True
        assert r.issues == ()

    def test_is_valid_with_warnings(self) -> None:
        r = ValidationResult(issues=(ValidationIssue(Severity.WARNING, "w"),))
        assert r.is_valid is True

    def test_is_valid_with_errors(self) -> None:
        r = ValidationResult(issues=(ValidationIssue(Severity.ERROR, "e"),))
        assert r.is_valid is False

    def test_errors_and_warnings(self) -> None:
        r = ValidationResult(
            issues=(
                ValidationIssue(Severity.ERROR, "e"),
                ValidationIssue(Severity.WARNING, "w"),
                ValidationIssue(Severity.INFO, "i"),
            )
        )
        assert len(r.errors()) == 1
        assert len(r.warnings()) == 1

    def test_merge(self) -> None:
        a = ValidationResult(issues=(ValidationIssue(Severity.ERROR, "a"),))
        b = ValidationResult(issues=(ValidationIssue(Severity.WARNING, "b"),))
        merged = ValidationResult.merge(a, b)
        assert len(merged.issues) == 2
        assert merged.is_valid is False

    def test_merge_empty(self) -> None:
        merged = ValidationResult.merge()
        assert merged.is_valid is True

    def test_frozen(self) -> None:
        r = ValidationResult.ok()
        with pytest.raises(AttributeError):
            r.issues = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ContentValidator
# ---------------------------------------------------------------------------


class TestContentValidator:
    def setup_method(self) -> None:
        self.v = ContentValidator()

    def test_valid_card(self) -> None:
        r = self.v.validate(front="What is Python?", back="A programming language.")
        assert r.is_valid is True

    def test_empty_front(self) -> None:
        r = self.v.validate(front="", back="An answer here.")
        assert r.is_valid is False
        assert any("empty" in i.message.lower() for i in r.errors())

    def test_empty_back(self) -> None:
        r = self.v.validate(front="What is Python?", back="  ")
        assert r.is_valid is False

    def test_short_front(self) -> None:
        r = self.v.validate(front="Hi?", back="A reasonable answer here.")
        assert any(i.severity == Severity.WARNING for i in r.issues)

    def test_long_back(self) -> None:
        r = self.v.validate(front="What is Python?", back="x" * 6000)
        assert any("5000" in i.message for i in r.issues)

    def test_unmatched_code_fence(self) -> None:
        r = self.v.validate(front="What?", back="```python\ncode")
        assert r.is_valid is False
        assert any("code fence" in i.message.lower() for i in r.issues)

    def test_matched_code_fence(self) -> None:
        r = self.v.validate(front="What is this?", back="```python\ncode\n```")
        assert r.is_valid is True


# ---------------------------------------------------------------------------
# FormatValidator
# ---------------------------------------------------------------------------


class TestFormatValidator:
    def setup_method(self) -> None:
        self.v = FormatValidator()

    def test_clean(self) -> None:
        r = self.v.validate(front="What?", back="Answer.")
        assert r.is_valid is True
        assert r.issues == ()

    def test_trailing_whitespace(self) -> None:
        r = self.v.validate(front="What? ", back="Answer.")
        assert any("trailing" in i.message.lower() for i in r.issues)

    def test_consecutive_blank_lines(self) -> None:
        r = self.v.validate(front="What?", back="A\n\n\nB")
        assert any("blank" in i.message.lower() for i in r.issues)


# ---------------------------------------------------------------------------
# HTMLValidator
# ---------------------------------------------------------------------------


class TestHTMLValidator:
    def setup_method(self) -> None:
        self.v = HTMLValidator()

    def test_no_html(self) -> None:
        r = self.v.validate(front="What?", back="Answer")
        assert r.is_valid is True

    def test_balanced_tags(self) -> None:
        r = self.v.validate(front="<b>bold</b>?", back="<i>ok</i>")
        assert r.is_valid is True

    def test_unclosed_tag(self) -> None:
        r = self.v.validate(front="<b>bold?", back="Answer")
        assert r.is_valid is False
        assert any("unclosed" in i.message.lower() for i in r.issues)

    def test_forbidden_tag(self) -> None:
        r = self.v.validate(front="What?", back="<script>alert(1)</script>")
        assert r.is_valid is False
        assert any("forbidden" in i.message.lower() for i in r.issues)

    def test_void_elements_ok(self) -> None:
        r = self.v.validate(front="What?", back="Line<br>break")
        assert r.is_valid is True

    def test_unexpected_closing(self) -> None:
        r = self.v.validate(front="What?", back="</div>text")
        assert r.is_valid is False


# ---------------------------------------------------------------------------
# TagValidator
# ---------------------------------------------------------------------------


class TestTagValidator:
    def setup_method(self) -> None:
        self.v = TagValidator()

    def test_valid_tags(self) -> None:
        r = self.v.validate(front="What?", back="Answer", tags=("python", "basics"))
        assert r.is_valid is True

    def test_empty_tag(self) -> None:
        r = self.v.validate(front="What?", back="Answer", tags=("python", ""))
        assert r.is_valid is False

    def test_invalid_chars(self) -> None:
        r = self.v.validate(front="What?", back="Answer", tags=("hello world",))
        assert any("invalid" in i.message.lower() for i in r.issues)

    def test_too_many_tags(self) -> None:
        tags = tuple(f"tag{i}" for i in range(25))
        r = self.v.validate(front="What?", back="Answer", tags=tags)
        assert any("too many" in i.message.lower() for i in r.issues)

    def test_duplicate_tags(self) -> None:
        r = self.v.validate(front="What?", back="Answer", tags=("a", "b", "a"))
        assert any("duplicate" in i.message.lower() for i in r.issues)

    def test_no_tags(self) -> None:
        r = self.v.validate(front="What?", back="Answer", tags=())
        assert r.is_valid is True


# ---------------------------------------------------------------------------
# ValidationPipeline
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    def test_empty_pipeline(self) -> None:
        p = ValidationPipeline([])
        r = p.run(front="What?", back="Answer")
        assert r.is_valid is True

    def test_single_validator(self) -> None:
        p = ValidationPipeline([ContentValidator()])
        r = p.run(front="What is Python?", back="A programming language.")
        assert r.is_valid is True

    def test_chain_validators(self) -> None:
        p = ValidationPipeline([ContentValidator(), FormatValidator(), TagValidator()])
        r = p.run(front="What is Python?", back="A programming language.", tags=("python",))
        assert r.is_valid is True

    def test_pipeline_aggregates_errors(self) -> None:
        p = ValidationPipeline([ContentValidator(), HTMLValidator()])
        r = p.run(front="", back="<script>x</script>")
        errors = r.errors()
        assert len(errors) >= 2  # empty front + forbidden tag

    def test_pipeline_passes_tags(self) -> None:
        p = ValidationPipeline([TagValidator()])
        r = p.run(front="What?", back="Answer", tags=("", "good"))
        assert r.is_valid is False


# ---------------------------------------------------------------------------
# QualityScore
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_creation(self) -> None:
        qs = QualityScore(
            clarity=0.8,
            atomicity=0.9,
            testability=0.7,
            memorability=0.6,
            accuracy=1.0,
        )
        assert qs.clarity == 0.8
        assert qs.overall == pytest.approx(0.8)

    def test_frozen(self) -> None:
        qs = QualityScore(
            clarity=1.0,
            atomicity=1.0,
            testability=1.0,
            memorability=1.0,
            accuracy=1.0,
        )
        with pytest.raises(AttributeError):
            qs.clarity = 0.5  # type: ignore[misc]

    def test_perfect_score(self) -> None:
        qs = QualityScore(
            clarity=1.0,
            atomicity=1.0,
            testability=1.0,
            memorability=1.0,
            accuracy=1.0,
        )
        assert qs.overall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# assess_quality
# ---------------------------------------------------------------------------


class TestAssessQuality:
    def test_good_card_scores_high(self) -> None:
        qs = assess_quality(
            front="What is the time complexity of binary search?",
            back="O(log n)",
        )
        assert qs.overall > 0.7

    def test_vague_card_scores_lower(self) -> None:
        qs = assess_quality(
            front="Explain everything about databases",
            back="Databases store data.",
        )
        assert qs.clarity < 0.8

    def test_yes_no_penalty(self) -> None:
        qs = assess_quality(
            front="Is Python interpreted?",
            back="Yes, Python is an interpreted language.",
        )
        good = assess_quality(
            front="What type of language is Python?",
            back="Python is an interpreted language.",
        )
        assert qs.clarity < good.clarity

    def test_long_answer_penalty(self) -> None:
        qs = assess_quality(
            front="What is Python?",
            back=" ".join(["word"] * 250),
        )
        assert qs.testability < 0.8
        assert qs.memorability < 0.8

    def test_multi_concept_atomicity(self) -> None:
        qs = assess_quality(
            front="What is Python and how does it compare to Java and C++?",
            back="Python is interpreted, Java and C++ are compiled.",
        )
        assert qs.atomicity < 0.8

    def test_empty_front(self) -> None:
        qs = assess_quality(front="", back="Answer here.")
        assert qs.accuracy < 0.8

    def test_enumeration_penalty(self) -> None:
        items = "\n".join(f"- item {i}" for i in range(10))
        qs = assess_quality(front="What are the items?", back=items)
        assert qs.memorability < 0.8
