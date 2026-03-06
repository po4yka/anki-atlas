"""Tests for packages.anki.connect: AnkiConnect async HTTP client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from packages.anki.connect import (
    ANKI_CONNECT_URL,
    ANKI_CONNECT_VERSION,
    AnkiConnectClient,
)
from packages.common.exceptions import AnkiConnectError


def _mock_transport(
    handler: Any,
) -> httpx.MockTransport:
    """Create a mock transport from a handler function."""
    return httpx.MockTransport(handler)


def _ok_response(result: Any = None) -> httpx.Response:
    """Build a successful AnkiConnect JSON response."""
    return httpx.Response(200, json={"result": result, "error": None})


def _error_response(error: str) -> httpx.Response:
    """Build an AnkiConnect error JSON response."""
    return httpx.Response(200, json={"result": None, "error": error})


# ---------------------------------------------------------------------------
# invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    async def test_invoke_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = request.content
            import json

            data = json.loads(body)
            assert data["action"] == "version"
            assert data["version"] == ANKI_CONNECT_VERSION
            return _ok_response(6)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.invoke("version")
        assert result == 6
        await client.close()

    async def test_invoke_with_params(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["params"] == {"deck": "Test"}
            return _ok_response(123)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.invoke("createDeck", deck="Test")
        assert result == 123
        await client.close()

    async def test_invoke_anki_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response("model was not found")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        with pytest.raises(AnkiConnectError, match="model was not found"):
            await client.invoke("addNote")
        await client.close()

    async def test_invoke_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        with pytest.raises(AnkiConnectError, match="HTTP error"):
            await client.invoke("version")
        await client.close()

    async def test_invoke_connect_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        with pytest.raises(AnkiConnectError, match="Failed to connect"):
            await client.invoke("version")
        await client.close()

    async def test_error_context_includes_action(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response("bad")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        with pytest.raises(AnkiConnectError) as exc_info:
            await client.invoke("testAction")
        assert exc_info.value.context["action"] == "testAction"
        await client.close()


# ---------------------------------------------------------------------------
# Context manager & lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_context_manager(self) -> None:
        async with AnkiConnectClient() as client:
            assert client._client is not None
        assert client._client is None

    async def test_lazy_client_creation(self) -> None:
        client = AnkiConnectClient()
        assert client._client is None
        inner = await client._get_client()
        assert inner is not None
        await client.close()

    async def test_default_url(self) -> None:
        client = AnkiConnectClient()
        assert client.url == ANKI_CONNECT_URL

    async def test_custom_url(self) -> None:
        client = AnkiConnectClient(url="http://example.com:9999")
        assert client.url == "http://example.com:9999"


# ---------------------------------------------------------------------------
# ping & version
# ---------------------------------------------------------------------------


class TestPingVersion:
    async def test_ping_success(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(6)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        assert await client.ping() is True
        await client.close()

    async def test_ping_failure(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        assert await client.ping() is False
        await client.close()

    async def test_version_success(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(6)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        assert await client.version() == 6
        await client.close()

    async def test_version_unreachable(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        assert await client.version() is None
        await client.close()


# ---------------------------------------------------------------------------
# Deck operations
# ---------------------------------------------------------------------------


class TestDeckOps:
    async def test_deck_names(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(["Default", "Kotlin"])

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.deck_names()
        assert result == ["Default", "Kotlin"]
        await client.close()

    async def test_create_deck(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(1234567890)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        deck_id = await client.create_deck("NewDeck")
        assert deck_id == 1234567890
        await client.close()

    async def test_delete_decks_empty(self) -> None:
        client = AnkiConnectClient()
        # Should not raise even without a transport
        await client.delete_decks([])

    async def test_delete_decks(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["params"]["decks"] == ["OldDeck"]
            assert data["params"]["cardsToo"] is True
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.delete_decks(["OldDeck"])
        await client.close()


# ---------------------------------------------------------------------------
# Note operations
# ---------------------------------------------------------------------------


class TestNoteOps:
    async def test_find_notes(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response([111, 222, 333])

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.find_notes("deck:Test")
        assert result == [111, 222, 333]
        await client.close()

    async def test_notes_info_empty(self) -> None:
        client = AnkiConnectClient()
        result = await client.notes_info([])
        assert result == []

    async def test_notes_info(self) -> None:
        note_data = [{"noteId": 1, "modelName": "Basic", "tags": [], "fields": {}}]

        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(note_data)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.notes_info([1])
        assert result == note_data
        await client.close()

    async def test_add_note(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            note = data["params"]["note"]
            assert note["deckName"] == "Test"
            assert note["modelName"] == "Basic"
            assert note["fields"] == {"Front": "Q", "Back": "A"}
            return _ok_response(999)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.add_note("Test", "Basic", {"Front": "Q", "Back": "A"})
        assert result == 999
        await client.close()

    async def test_add_note_duplicate_returns_none(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response("cannot create note because it is a duplicate")

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.add_note("Test", "Basic", {"Front": "Q", "Back": "A"})
        assert result is None
        await client.close()

    async def test_update_note_fields(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["params"]["note"]["id"] == 42
            assert data["params"]["note"]["fields"] == {"Front": "Updated"}
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.update_note_fields(42, {"Front": "Updated"})
        await client.close()

    async def test_delete_notes_empty(self) -> None:
        client = AnkiConnectClient()
        await client.delete_notes([])

    async def test_delete_notes(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["params"]["notes"] == [1, 2, 3]
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.delete_notes([1, 2, 3])
        await client.close()


# ---------------------------------------------------------------------------
# Tag operations
# ---------------------------------------------------------------------------


class TestTagOps:
    async def test_get_tags(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(["tag1", "tag2"])

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.get_tags()
        assert result == ["tag1", "tag2"]
        await client.close()

    async def test_add_tags_empty_notes(self) -> None:
        client = AnkiConnectClient()
        await client.add_tags([], "tag1")

    async def test_add_tags_empty_tags(self) -> None:
        client = AnkiConnectClient()
        await client.add_tags([1], "")

    async def test_add_tags(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["params"]["notes"] == [1, 2]
            assert data["params"]["tags"] == "tag1 tag2"
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.add_tags([1, 2], "tag1 tag2")
        await client.close()

    async def test_remove_tags(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.remove_tags([1], "old_tag")
        await client.close()


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------


class TestModelOps:
    async def test_model_names(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(["Basic", "Cloze"])

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.model_names()
        assert result == ["Basic", "Cloze"]
        await client.close()

    async def test_model_field_names(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _ok_response(["Front", "Back"])

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        result = await client.model_field_names("Basic")
        assert result == ["Front", "Back"]
        await client.close()


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


class TestSync:
    async def test_sync(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            data = json.loads(request.content)
            assert data["action"] == "sync"
            return _ok_response(None)

        client = AnkiConnectClient()
        client._client = httpx.AsyncClient(transport=_mock_transport(handler))
        await client.sync()
        await client.close()
