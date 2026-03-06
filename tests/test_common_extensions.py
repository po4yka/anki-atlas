"""Tests for common package type and exception extensions."""

from __future__ import annotations

from packages.common import (
    AnkiAtlasError,
    AnkiConnectError,
    CardGenerationError,
    CardId,
    CardValidationError,
    DeckName,
    Language,
    NoteId,
    ObsidianParseError,
    ProviderError,
    SlugStr,
    SyncConflictError,
)


class TestLanguageEnum:
    def test_has_ten_members(self) -> None:
        assert len(Language) == 10

    def test_en_value(self) -> None:
        assert Language.EN == "en"

    def test_membership(self) -> None:
        assert "ru" in Language.__members__.values()

    def test_is_str(self) -> None:
        assert isinstance(Language.JA, str)


class TestNewTypes:
    def test_slug_str(self) -> None:
        s = SlugStr("hello-world")
        assert s == "hello-world"

    def test_card_id(self) -> None:
        cid = CardId(42)
        assert cid == 42

    def test_note_id(self) -> None:
        nid = NoteId(99)
        assert nid == 99

    def test_deck_name(self) -> None:
        dn = DeckName("My::Deck")
        assert dn == "My::Deck"


class TestNewExceptions:
    NEW_EXCEPTIONS = (
        CardGenerationError,
        CardValidationError,
        ProviderError,
        ObsidianParseError,
        SyncConflictError,
        AnkiConnectError,
    )

    def test_all_inherit_from_base(self) -> None:
        for exc_cls in self.NEW_EXCEPTIONS:
            assert issubclass(exc_cls, AnkiAtlasError)

    def test_context_support(self) -> None:
        for exc_cls in self.NEW_EXCEPTIONS:
            err = exc_cls("test", context={"key": "val"})
            assert err.context == {"key": "val"}

    def test_default_empty_context(self) -> None:
        for exc_cls in self.NEW_EXCEPTIONS:
            err = exc_cls("test")
            assert err.context == {}

    def test_raiseable(self) -> None:
        for exc_cls in self.NEW_EXCEPTIONS:
            try:
                raise exc_cls("boom")
            except AnkiAtlasError as e:
                assert str(e) == "boom"
