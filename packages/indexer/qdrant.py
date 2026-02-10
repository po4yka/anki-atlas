"""Qdrant vector database repository."""

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from packages.common.config import Settings, get_settings

# Collection name for Anki notes
COLLECTION_NAME = "anki_notes"


@dataclass
class NotePayload:
    """Payload stored with each vector in Qdrant."""

    note_id: int
    deck_names: list[str]
    tags: list[str]
    model_id: int
    content_hash: str
    # Optional learning stats
    mature: bool = False
    lapses: int = 0
    reps: int = 0
    fail_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Qdrant."""
        return {
            "note_id": self.note_id,
            "deck_names": self.deck_names,
            "tags": self.tags,
            "model_id": self.model_id,
            "content_hash": self.content_hash,
            "mature": self.mature,
            "lapses": self.lapses,
            "reps": self.reps,
            "fail_rate": self.fail_rate,
        }


@dataclass
class UpsertResult:
    """Result of an upsert operation."""

    upserted: int = 0
    skipped: int = 0


class QdrantRepository:
    """Repository for Qdrant vector operations."""

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Initialize Qdrant repository.

        Args:
            settings: Application settings.
            collection_name: Name of the collection to use.
        """
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._client: AsyncQdrantClient | None = None

    async def get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.settings.qdrant_url)
        return self._client

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ensure_collection(self, dimension: int) -> bool:
        """Ensure the collection exists with proper schema.

        Args:
            dimension: Vector dimension.

        Returns:
            True if collection was created, False if it already existed.
        """
        client = await self.get_client()

        # Check if collection exists
        collections = await client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if exists:
            return False

        # Create collection with optimized settings
        await client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
            # Optimize for filtering
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
        )

        # Create payload indexes for filtering
        await client.create_payload_index(
            collection_name=self.collection_name,
            field_name="note_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        await client.create_payload_index(
            collection_name=self.collection_name,
            field_name="deck_names",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=self.collection_name,
            field_name="tags",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=self.collection_name,
            field_name="model_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        await client.create_payload_index(
            collection_name=self.collection_name,
            field_name="mature",
            field_schema=models.PayloadSchemaType.BOOL,
        )

        return True

    async def get_existing_hashes(self, note_ids: list[int]) -> dict[int, str]:
        """Get content hashes for existing notes.

        Args:
            note_ids: List of note IDs to check.

        Returns:
            Dictionary mapping note_id to content_hash.
        """
        if not note_ids:
            return {}

        client = await self.get_client()
        result: dict[int, str] = {}

        # Scroll through matching points
        offset = None
        while True:
            records, offset = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="note_id",
                            match=models.MatchAny(any=note_ids),
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=["note_id", "content_hash"],
                with_vectors=False,
            )

            for record in records:
                if record.payload:
                    note_id = record.payload.get("note_id")
                    content_hash = record.payload.get("content_hash")
                    if note_id is not None and content_hash is not None:
                        result[note_id] = content_hash

            if offset is None:
                break

        return result

    async def upsert_vectors(
        self,
        vectors: list[list[float]],
        payloads: list[NotePayload],
    ) -> int:
        """Upsert vectors with payloads.

        Uses note_id as point ID for idempotent updates.

        Args:
            vectors: List of embedding vectors.
            payloads: List of payloads (same length as vectors).

        Returns:
            Number of points upserted.
        """
        if not vectors or len(vectors) != len(payloads):
            return 0

        client = await self.get_client()

        points = [
            models.PointStruct(
                id=payload.note_id,
                vector=vector,
                payload=payload.to_dict(),
            )
            for vector, payload in zip(vectors, payloads, strict=True)
        ]

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    async def delete_vectors(self, note_ids: list[int]) -> int:
        """Delete vectors by note IDs.

        Args:
            note_ids: List of note IDs to delete.

        Returns:
            Number of points deleted.
        """
        if not note_ids:
            return 0

        client = await self.get_client()

        await client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=note_ids,
            ),
        )

        return len(note_ids)

    async def get_collection_info(self) -> dict[str, Any] | None:
        """Get collection information.

        Returns:
            Collection info or None if collection doesn't exist.
        """
        client = await self.get_client()

        try:
            info = await client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception:
            return None

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable and healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            client = await self.get_client()
            # Simple health check - get collections list
            await client.get_collections()
            return True
        except Exception:
            return False


# Module-level singleton
_repository: QdrantRepository | None = None


async def get_qdrant_repository(
    settings: Settings | None = None,
) -> QdrantRepository:
    """Get or create the Qdrant repository singleton.

    Args:
        settings: Application settings.

    Returns:
        QdrantRepository instance.
    """
    global _repository
    if _repository is None:
        _repository = QdrantRepository(settings)
    return _repository


async def close_qdrant_repository() -> None:
    """Close the Qdrant repository singleton."""
    global _repository
    if _repository is not None:
        await _repository.close()
        _repository = None
