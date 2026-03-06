"""Tests for packages.card domain models and slug service."""

from __future__ import annotations

import pytest

from packages.card import Card, CardManifest, SlugService, SyncAction, SyncActionType
from packages.common.exceptions import CardValidationError

# -- Helpers ------------------------------------------------------------------


def _make_manifest(**overrides: object) -> CardManifest:
    defaults: dict[str, object] = {
        "slug": "python-decorators-1-en",
        "slug_base": "python-decorators-1",
        "lang": "en",
        "source_path": "notes/python/decorators.md",
        "source_anchor": "Q1-question",
        "note_id": "note-001",
        "note_title": "Python Decorators",
        "card_index": 0,
    }
    defaults.update(overrides)
    return CardManifest(**defaults)  # type: ignore[arg-type]


def _make_card(**overrides: object) -> Card:
    manifest_overrides = {}
    if "slug" in overrides:
        manifest_overrides["slug"] = overrides["slug"]
    if "language" in overrides:
        manifest_overrides["lang"] = overrides["language"]
    manifest = overrides.pop("manifest", None) or _make_manifest(**manifest_overrides)
    defaults: dict[str, object] = {
        "slug": "python-decorators-1-en",
        "language": "en",
        "apf_html": "<div>What is a decorator?</div><div>A decorator wraps a function.</div>",
        "manifest": manifest,
        "note_type": "APF::Simple",
    }
    defaults.update(overrides)
    return Card(**defaults)  # type: ignore[arg-type]


# -- CardManifest tests -------------------------------------------------------


class TestCardManifest:
    def test_create_valid(self) -> None:
        m = _make_manifest()
        assert m.slug == "python-decorators-1-en"
        assert m.lang == "en"
        assert m.card_index == 0

    def test_empty_slug_raises(self) -> None:
        with pytest.raises(CardValidationError, match="slug cannot be empty"):
            _make_manifest(slug="")

    def test_invalid_lang_raises(self) -> None:
        with pytest.raises(CardValidationError, match="not in valid languages"):
            _make_manifest(lang="xx")

    def test_negative_card_index_raises(self) -> None:
        with pytest.raises(CardValidationError, match="card_index must be >= 0"):
            _make_manifest(card_index=-1)

    def test_invalid_hash6_raises(self) -> None:
        with pytest.raises(CardValidationError, match="hash6 must be exactly 6"):
            _make_manifest(hash6="abc")

    def test_with_guid_immutability(self) -> None:
        m = _make_manifest()
        m2 = m.with_guid("guid-123")
        assert m.guid is None
        assert m2.guid == "guid-123"

    def test_with_hash_immutability(self) -> None:
        m = _make_manifest()
        m2 = m.with_hash("abcdef")
        assert m.hash6 is None
        assert m2.hash6 == "abcdef"

    def test_anchor_url(self) -> None:
        m = _make_manifest(source_path="vault/notes/python/decorators.md")
        assert m.anchor_url == "[[python/decorators.md#Q1-question]]"

    def test_is_linked_to_note(self) -> None:
        m = _make_manifest()
        assert m.is_linked_to_note is True

    def test_with_fsrs_metadata(self) -> None:
        m = _make_manifest()
        m2 = m.with_fsrs_metadata(difficulty=0.5, cognitive_load="basic")
        assert m2.difficulty == 0.5
        assert m2.cognitive_load == "basic"

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(CardValidationError, match="difficulty must be between"):
            _make_manifest(difficulty=1.5)

    def test_invalid_cognitive_load_raises(self) -> None:
        with pytest.raises(CardValidationError, match="cognitive_load must be one of"):
            _make_manifest(cognitive_load="expert")


# -- Card tests ----------------------------------------------------------------


class TestCard:
    def test_create_valid(self) -> None:
        card = _make_card()
        assert card.slug == "python-decorators-1-en"
        assert card.language == "en"
        assert card.is_new is True

    def test_empty_slug_raises(self) -> None:
        with pytest.raises(CardValidationError, match="slug cannot be empty"):
            _make_card(slug="")

    def test_invalid_language_raises(self) -> None:
        with pytest.raises(CardValidationError, match="not in valid languages"):
            _make_card(language="xx")

    def test_short_apf_html_raises(self) -> None:
        with pytest.raises(CardValidationError, match="too short"):
            _make_card(apf_html="<b>hi</b>")

    def test_invalid_note_type_raises(self) -> None:
        with pytest.raises(CardValidationError, match="not in valid types"):
            _make_card(note_type="Invalid")

    def test_content_hash_deterministic(self) -> None:
        card = _make_card()
        assert len(card.content_hash) == 6
        assert card.content_hash == card.content_hash

    def test_with_guid_immutability(self) -> None:
        card = _make_card()
        card2 = card.with_guid("guid-abc")
        assert card.anki_guid is None
        assert card2.anki_guid == "guid-abc"
        assert card2.manifest.guid == "guid-abc"
        assert card2.is_new is False

    def test_with_guid_empty_raises(self) -> None:
        card = _make_card()
        with pytest.raises(CardValidationError, match="guid cannot be empty"):
            card.with_guid("")

    def test_update_content(self) -> None:
        card = _make_card()
        new_html = "<div>Updated question</div><div>Updated answer with detail.</div>"
        card2 = card.update_content(new_html)
        assert card2.apf_html == new_html
        assert card2.manifest.hash6 == card2.content_hash

    def test_with_tags(self) -> None:
        card = _make_card()
        card2 = card.with_tags(["python", "basics"])
        assert card2.tags == ["python", "basics"]
        assert card.tags == []

    def test_manifest_lang_mismatch_raises(self) -> None:
        manifest = _make_manifest(lang="ru", slug="python-decorators-1-ru")
        with pytest.raises(CardValidationError, match="does not match"):
            Card(
                slug="python-decorators-1-ru",
                language="en",
                apf_html="<div>What is a decorator?</div><div>Wraps a function nicely.</div>",
                manifest=manifest,
                note_type="APF::Simple",
            )


# -- SyncAction tests ----------------------------------------------------------


class TestSyncAction:
    def test_create_action(self) -> None:
        card = _make_card()
        action = SyncAction(action_type=SyncActionType.CREATE, card=card)
        assert action.action_type == SyncActionType.CREATE
        assert action.is_destructive is False
        assert action.requires_confirmation is False

    def test_update_requires_guid(self) -> None:
        card = _make_card()
        with pytest.raises(CardValidationError, match="anki_guid is required"):
            SyncAction(action_type=SyncActionType.UPDATE, card=card)

    def test_delete_is_destructive(self) -> None:
        card = _make_card()
        action = SyncAction(
            action_type=SyncActionType.DELETE, card=card, anki_guid="guid-1"
        )
        assert action.is_destructive is True
        assert action.requires_confirmation is True

    def test_describe(self) -> None:
        card = _make_card()
        action = SyncAction(
            action_type=SyncActionType.SKIP, card=card, reason="up-to-date"
        )
        assert action.describe() == "SKIP: python-decorators-1-en (up-to-date)"

    def test_sync_action_type_values(self) -> None:
        assert SyncActionType.CREATE.value == "create"
        assert SyncActionType.UPDATE.value == "update"
        assert SyncActionType.DELETE.value == "delete"
        assert SyncActionType.SKIP.value == "skip"


# -- SlugService tests ---------------------------------------------------------


class TestSlugService:
    def test_slugify_basic(self) -> None:
        assert SlugService.slugify("Hello World") == "hello-world"

    def test_slugify_unicode(self) -> None:
        result = SlugService.slugify("Cafe Resume")
        assert result == "cafe-resume"

    def test_slugify_empty(self) -> None:
        assert SlugService.slugify("") == ""

    def test_slugify_special_chars(self) -> None:
        assert SlugService.slugify("Python 3.12 Features!") == "python-3-12-features"

    def test_generate_slug(self) -> None:
        slug = SlugService.generate_slug("Python", "decorators", 1, "en")
        assert slug == "python-decorators-1-en"

    def test_generate_slug_base(self) -> None:
        base = SlugService.generate_slug_base("Python", "decorators", 1)
        assert base == "python-decorators-1"

    def test_compute_hash_deterministic(self) -> None:
        h1 = SlugService.compute_hash("hello")
        h2 = SlugService.compute_hash("hello")
        assert h1 == h2
        assert len(h1) == 6

    def test_compute_hash_custom_length(self) -> None:
        h = SlugService.compute_hash("hello", length=12)
        assert len(h) == 12

    def test_is_valid_slug(self) -> None:
        assert SlugService.is_valid_slug("python-decorators-1-en") is True
        assert SlugService.is_valid_slug("Invalid Slug!") is False
        assert SlugService.is_valid_slug("ab") is False
        assert SlugService.is_valid_slug("no--double-hyphens-en") is False

    def test_compute_content_hash(self) -> None:
        h = SlugService.compute_content_hash("What is X?", "X is Y.")
        assert len(h) == 12

    def test_compute_metadata_hash(self) -> None:
        h = SlugService.compute_metadata_hash("APF::Simple", ["python", "basics"])
        assert len(h) == 6

    def test_generate_deterministic_guid(self) -> None:
        g1 = SlugService.generate_deterministic_guid("slug-1-en", "path/to/note.md")
        g2 = SlugService.generate_deterministic_guid("slug-1-en", "path/to/note.md")
        assert g1 == g2
        assert len(g1) == 32

    def test_extract_components(self) -> None:
        result = SlugService.extract_components("python-decorators-1-en")
        assert result["lang"] == "en"
        assert result["index"] == 1
