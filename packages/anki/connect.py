"""AnkiConnect HTTP client for async communication with Anki.

Provides a unified async client for all AnkiConnect API operations
including note CRUD, deck management, tags, and model queries.

Usage:
    async with AnkiConnectClient() as client:
        if await client.ping():
            decks = await client.deck_names()
"""

from __future__ import annotations

from typing import Any, Final, cast

import httpx
import structlog

from packages.common.exceptions import AnkiConnectError

logger = structlog.get_logger()

ANKI_CONNECT_URL: Final[str] = "http://localhost:8765"
ANKI_CONNECT_VERSION: Final[int] = 6
DEFAULT_TIMEOUT: Final[float] = 30.0


class AnkiConnectClient:
    """Async client for AnkiConnect API.

    Can be used as an async context manager or manually managed:

        async with AnkiConnectClient() as client:
            await client.ping()

        # or
        client = AnkiConnectClient()
        try:
            await client.ping()
        finally:
            await client.close()
    """

    def __init__(
        self,
        url: str = ANKI_CONNECT_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AnkiConnectClient:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def invoke(self, action: str, **params: Any) -> Any:
        """Send a request to AnkiConnect API.

        Args:
            action: The AnkiConnect action name.
            **params: Parameters for the action.

        Returns:
            The result from AnkiConnect.

        Raises:
            AnkiConnectError: If the request fails or AnkiConnect returns an error.
        """
        payload: dict[str, Any] = {
            "action": action,
            "version": ANKI_CONNECT_VERSION,
        }
        if params:
            payload["params"] = params

        try:
            client = await self._get_client()
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise AnkiConnectError(
                f"Failed to connect to AnkiConnect at {self.url}. "
                "Is Anki running with AnkiConnect installed?",
                context={"action": action},
            ) from e
        except httpx.HTTPError as e:
            raise AnkiConnectError(
                f"HTTP error communicating with AnkiConnect: {e}",
                context={"action": action},
            ) from e

        result = response.json()

        if result.get("error") is not None:
            raise AnkiConnectError(
                result["error"],
                context={"action": action},
            )

        return result.get("result")

    # -- Connection & Version --------------------------------------------------

    async def ping(self) -> bool:
        """Check if AnkiConnect is reachable."""
        try:
            version = await self.invoke("version")
            logger.debug("anki_connect_ping", version=version)
            return version is not None
        except AnkiConnectError:
            return False

    async def version(self) -> int | None:
        """Get the AnkiConnect version, or None if unreachable."""
        try:
            return cast("int", await self.invoke("version"))
        except AnkiConnectError:
            return None

    # -- Deck Operations -------------------------------------------------------

    async def deck_names(self) -> list[str]:
        """Get all deck names."""
        return cast("list[str]", await self.invoke("deckNames"))

    async def create_deck(self, deck_name: str) -> int:
        """Create a deck (no-op if it already exists). Returns deck ID."""
        return cast("int", await self.invoke("createDeck", deck=deck_name))

    async def delete_decks(
        self,
        deck_names: list[str],
        *,
        cards_too: bool = True,
    ) -> None:
        """Delete decks. cards_too must be True since Anki 2.1.28."""
        if not deck_names:
            return
        await self.invoke("deleteDecks", decks=deck_names, cardsToo=cards_too)

    # -- Note Operations -------------------------------------------------------

    async def find_notes(self, query: str) -> list[int]:
        """Find note IDs matching an Anki search query."""
        return cast("list[int]", await self.invoke("findNotes", query=query))

    async def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed info for notes. Returns raw API dicts."""
        if not note_ids:
            return []
        return cast("list[dict[str, Any]]", await self.invoke("notesInfo", notes=note_ids))

    async def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str] | None = None,
        *,
        allow_duplicate: bool = False,
    ) -> int | None:
        """Add a new note. Returns note ID, or None if duplicate."""
        note: dict[str, Any] = {
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "tags": tags or [],
            "options": {"allowDuplicate": allow_duplicate},
        }
        try:
            return cast("int", await self.invoke("addNote", note=note))
        except AnkiConnectError as e:
            if "duplicate" in str(e).lower():
                return None
            raise

    async def update_note_fields(
        self,
        note_id: int,
        fields: dict[str, str],
    ) -> None:
        """Update fields of an existing note."""
        await self.invoke(
            "updateNoteFields",
            note={"id": note_id, "fields": fields},
        )

    async def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes by ID."""
        if not note_ids:
            return
        await self.invoke("deleteNotes", notes=note_ids)

    # -- Tag Operations --------------------------------------------------------

    async def get_tags(self) -> list[str]:
        """Get all tags from the collection."""
        return cast("list[str]", await self.invoke("getTags"))

    async def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Add space-separated tags to notes."""
        if not note_ids or not tags:
            return
        await self.invoke("addTags", notes=note_ids, tags=tags)

    async def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remove space-separated tags from notes."""
        if not note_ids or not tags:
            return
        await self.invoke("removeTags", notes=note_ids, tags=tags)

    # -- Model Operations ------------------------------------------------------

    async def model_names(self) -> list[str]:
        """Get all note type (model) names."""
        return cast("list[str]", await self.invoke("modelNames"))

    async def model_field_names(self, model_name: str) -> list[str]:
        """Get field names for a note type."""
        return cast("list[str]", await self.invoke("modelFieldNames", modelName=model_name))

    # -- Sync ------------------------------------------------------------------

    async def sync(self) -> None:
        """Trigger Anki sync."""
        await self.invoke("sync")
