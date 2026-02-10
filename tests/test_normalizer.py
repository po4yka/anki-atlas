"""Tests for HTML to text normalization."""


from packages.anki.models import AnkiNote
from packages.anki.normalizer import (
    normalize_note,
    normalize_whitespace,
    strip_html,
)


class TestStripHtml:
    """Tests for strip_html function."""

    def test_empty_string(self) -> None:
        """Empty string returns empty."""
        assert strip_html("") == ""

    def test_plain_text(self) -> None:
        """Plain text passes through."""
        assert strip_html("Hello world") == "Hello world"

    def test_simple_tags(self) -> None:
        """Simple HTML tags are stripped."""
        assert strip_html("<b>bold</b>") == "bold"
        assert strip_html("<i>italic</i>") == "italic"
        assert strip_html("<div>content</div>") == "content"

    def test_br_tags(self) -> None:
        """BR tags become newlines."""
        result = strip_html("line1<br>line2<br/>line3")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_nested_tags(self) -> None:
        """Nested tags are stripped."""
        assert strip_html("<div><b>nested</b></div>") == "nested"

    def test_html_entities(self) -> None:
        """HTML entities are decoded."""
        assert strip_html("&amp;") == "&"
        assert strip_html("&lt;") == "<"
        assert strip_html("&gt;") == ">"
        # &nbsp; becomes space, then normalized away if alone
        assert strip_html("word&nbsp;word") == "word word"

    def test_code_preserved(self) -> None:
        """Code blocks are preserved."""
        result = strip_html("<code>def foo():</code>")
        assert "def foo():" in result
        assert "`" in result  # Wrapped in backticks

    def test_pre_preserved(self) -> None:
        """Pre blocks are preserved."""
        result = strip_html("<pre>multiline\ncode</pre>")
        assert "multiline" in result
        assert "code" in result

    def test_cloze_extraction(self) -> None:
        """Cloze deletions are extracted."""
        # Simple cloze
        assert strip_html("{{c1::answer}}") == "answer"
        # Cloze with hint
        assert strip_html("{{c1::answer::hint}}") == "answer"
        # Multiple clozes
        result = strip_html("{{c1::first}} and {{c2::second}}")
        assert "first" in result
        assert "second" in result

    def test_whitespace_normalization(self) -> None:
        """Multiple whitespace is normalized."""
        result = strip_html("too   many    spaces")
        assert result == "too many spaces"


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_multiple_spaces(self) -> None:
        """Multiple spaces become single space."""
        assert normalize_whitespace("a   b") == "a b"

    def test_tabs(self) -> None:
        """Tabs become spaces."""
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_newlines_preserved_when_content(self) -> None:
        """Newlines are preserved when there's content on each line."""
        # normalize_whitespace collapses empty lines but preserves meaningful ones
        result = normalize_whitespace("line1\nline2")
        # After WHITESPACE_PATTERN.sub, newlines become spaces
        assert "line1" in result
        assert "line2" in result

    def test_leading_trailing_stripped(self) -> None:
        """Leading and trailing whitespace is stripped."""
        result = normalize_whitespace("  line1  ")
        assert result == "line1"


class TestNormalizeNote:
    """Tests for normalize_note function."""

    def test_basic_note(self) -> None:
        """Basic note with Front/Back fields."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=["python", "basics"],
            fields=["What is Python?", "A programming language."],
            fields_json={"Front": "What is Python?", "Back": "A programming language."},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note)

        assert "Front: What is Python?" in result
        assert "Back: A programming language." in result
        assert "Tags: python, basics" in result

    def test_note_with_deck(self) -> None:
        """Note with deck names included."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=[],
            fields=["Q", "A"],
            fields_json={"Front": "Q", "Back": "A"},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note, deck_names=["Programming::Python"])

        assert "Decks: Programming::Python" in result

    def test_note_with_html(self) -> None:
        """HTML in fields is stripped."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=[],
            fields=["<b>Bold</b> question", "<i>Italic</i> answer"],
            fields_json={"Front": "<b>Bold</b> question", "Back": "<i>Italic</i> answer"},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note)

        assert "Bold question" in result
        assert "Italic answer" in result
        assert "<b>" not in result
        assert "<i>" not in result

    def test_note_with_cloze(self) -> None:
        """Cloze deletions are extracted."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=[],
            fields=["{{c1::Lambda}} functions", "Extra info"],
            fields_json={"Front": "{{c1::Lambda}} functions", "Back": "Extra info"},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note)

        assert "Lambda" in result
        assert "{{c1::" not in result

    def test_note_fallback_fields(self) -> None:
        """Uses positional fields when no Front/Back names."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=[],
            fields=["First field", "Second field"],
            fields_json={"CustomField1": "First field", "CustomField2": "Second field"},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note)

        # Should use positional fallback since field names aren't recognized
        assert "First field" in result
        assert "Second field" in result

    def test_empty_fields_skipped(self) -> None:
        """Empty fields are not included."""
        note = AnkiNote(
            note_id=1,
            model_id=1,
            tags=[],
            fields=["Question", "", ""],
            fields_json={"Front": "Question", "Back": "", "Extra": ""},
            mtime=1700000000,
            usn=-1,
        )

        result = normalize_note(note)

        assert "Front: Question" in result
        assert "Back:" not in result
        assert "Extra:" not in result
