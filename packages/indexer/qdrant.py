"""Qdrant vector database repository."""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import AsyncQdrantClient, models

from packages.common.config import Settings, get_settings
from packages.common.logging import get_logger

logger = get_logger(module=__name__)

# Collection name for Anki notes
COLLECTION_NAME = "anki_notes"
DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "sparse"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DimensionMismatchError(Exception):
    """Raised when requested dimension doesn't match existing collection."""

    def __init__(self, collection: str, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Collection '{collection}' has dimension {actual}, "
            f"but provider requires {expected}. "
            f"Use --force-reindex to recreate the collection."
        )


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
        self._supports_sparse: bool | None = None

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
            self._supports_sparse = None

    @staticmethod
    def text_to_sparse_vector(text: str) -> models.SparseVector:
        """Convert text into a hashed sparse vector for Qdrant sparse retrieval."""
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        token_counts = Counter(tokens)
        index_weights: dict[int, float] = {}
        for token, count in token_counts.items():
            token_hash = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            index = int.from_bytes(token_hash, byteorder="big", signed=False)
            weight = 1.0 + math.log(float(count))
            index_weights[index] = index_weights.get(index, 0.0) + weight

        norm = math.sqrt(sum(w * w for w in index_weights.values()))
        if norm > 0:
            for index in list(index_weights.keys()):
                index_weights[index] /= norm

        sorted_pairs = sorted(index_weights.items(), key=lambda item: item[0])
        return models.SparseVector(
            indices=[idx for idx, _ in sorted_pairs],
            values=[float(val) for _, val in sorted_pairs],
        )

    @staticmethod
    def _collection_has_sparse(info: models.CollectionInfo) -> bool:
        """Check whether collection config includes configured sparse vectors."""
        sparse_cfg = info.config.params.sparse_vectors
        return isinstance(sparse_cfg, dict) and SPARSE_VECTOR_NAME in sparse_cfg

    async def _collection_supports_sparse(self, *, refresh: bool = False) -> bool:
        """Return whether current collection is configured with sparse vectors."""
        if self._supports_sparse is not None and not refresh:
            return self._supports_sparse

        client = await self.get_client()
        info = await client.get_collection(self.collection_name)
        self._supports_sparse = self._collection_has_sparse(info)
        return self._supports_sparse

    async def ensure_collection(self, dimension: int) -> bool:
        """Ensure the collection exists with proper schema.

        Args:
            dimension: Vector dimension.

        Returns:
            True if collection was created, False if it already existed.

        Raises:
            DimensionMismatchError: If collection exists with a different dimension.
        """
        client = await self.get_client()

        # Check if collection exists
        collections = await client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if exists:
            # Validate dimension matches
            info = await client.get_collection(self.collection_name)
            self._supports_sparse = self._collection_has_sparse(info)

            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, models.VectorParams):
                actual_dim = vectors_config.size
            elif isinstance(vectors_config, dict):
                dense_params = vectors_config.get("dense")
                if dense_params is None:
                    dense_params = next(iter(vectors_config.values()), None)
                if not isinstance(dense_params, models.VectorParams):
                    raise DimensionMismatchError(self.collection_name, dimension, -1)
                actual_dim = dense_params.size
            else:
                actual_dim = -1

            if actual_dim != dimension:
                raise DimensionMismatchError(self.collection_name, dimension, actual_dim)

            if not self._supports_sparse:
                logger.warning(
                    "qdrant_sparse_not_configured",
                    collection=self.collection_name,
                    action="fallback_dense_only",
                )
            return False

        await self._create_collection(client, dimension)
        self._supports_sparse = True
        return True

    async def recreate_collection(self, dimension: int) -> None:
        """Drop and recreate the collection with a new dimension.

        Args:
            dimension: Vector dimension for the new collection.
        """
        client = await self.get_client()

        collections = await client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        if exists:
            logger.warning(
                "recreating_collection",
                collection=self.collection_name,
                dimension=dimension,
            )
            await client.delete_collection(self.collection_name)

        await self._create_collection(client, dimension)
        self._supports_sparse = True

    async def _create_collection(self, client: AsyncQdrantClient, dimension: int) -> None:
        """Create the collection with optimized settings and payload indexes."""
        # Configure quantization for memory optimization
        quantization_config: models.ScalarQuantization | models.BinaryQuantization | None = None
        if self.settings.qdrant_quantization == "scalar":
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                )
            )
        elif self.settings.qdrant_quantization == "binary":
            quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,
                )
            )

        # Configure on-disk storage for large collections
        vectors_config = models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE,
            on_disk=self.settings.qdrant_on_disk,
        )

        logger.info(
            "creating_collection",
            collection=self.collection_name,
            dimension=dimension,
            quantization=self.settings.qdrant_quantization,
            on_disk=self.settings.qdrant_on_disk,
        )

        await client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            quantization_config=quantization_config,
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
        sparse_vectors: list[models.SparseVector] | None = None,
    ) -> int:
        """Upsert vectors with payloads.

        Uses note_id as point ID for idempotent updates.

        Args:
            vectors: List of embedding vectors.
            payloads: List of payloads (same length as vectors).
            sparse_vectors: Optional sparse vectors (same length as vectors).

        Returns:
            Number of points upserted.
        """
        if not vectors or len(vectors) != len(payloads):
            return 0
        if sparse_vectors is not None and len(sparse_vectors) != len(vectors):
            raise ValueError("sparse_vectors length must match vectors length")

        client = await self.get_client()
        use_sparse_vectors = sparse_vectors is not None and await self._collection_supports_sparse()

        if use_sparse_vectors:
            points = [
                models.PointStruct(
                    id=payload.note_id,
                    vector={
                        DENSE_VECTOR_NAME: vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=payload.to_dict(),
                )
                for vector, sparse_vector, payload in zip(
                    vectors, sparse_vectors or [], payloads, strict=True
                )
            ]
        else:
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

    async def search(
        self,
        query_vector: list[float],
        query_sparse_vector: models.SparseVector | None = None,
        limit: int = 50,
        prefetch_limit: int | None = None,
        deck_names: list[str] | None = None,
        deck_names_exclude: list[str] | None = None,
        tags: list[str] | None = None,
        tags_exclude: list[str] | None = None,
        model_ids: list[int] | None = None,
        mature_only: bool = False,
        max_lapses: int | None = None,
        min_reps: int | None = None,
    ) -> list[tuple[int, float]]:
        """Search for similar vectors with optional filters.

        Args:
            query_vector: Query embedding vector.
            query_sparse_vector: Optional sparse query vector.
            limit: Maximum number of results.
            prefetch_limit: Candidate pool size per prefetch branch.
            deck_names: Filter by deck names (any match).
            deck_names_exclude: Exclude notes in these decks.
            tags: Filter by tags (any match).
            tags_exclude: Exclude notes with these tags.
            model_ids: Filter by model IDs.
            mature_only: Only return mature cards (ivl >= 21).
            max_lapses: Maximum lapses filter.
            min_reps: Minimum reps filter.

        Returns:
            List of (note_id, score) tuples ordered by similarity.
        """
        client = await self.get_client()

        # Build filter conditions
        must_conditions: list[models.Condition] = []
        must_not_conditions: list[models.Condition] = []

        if deck_names:
            must_conditions.append(
                models.FieldCondition(
                    key="deck_names",
                    match=models.MatchAny(any=deck_names),
                )
            )

        if tags:
            must_conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=tags),
                )
            )

        if deck_names_exclude:
            must_not_conditions.append(
                models.FieldCondition(
                    key="deck_names",
                    match=models.MatchAny(any=deck_names_exclude),
                )
            )

        if tags_exclude:
            must_not_conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=tags_exclude),
                )
            )

        if model_ids:
            must_conditions.append(
                models.FieldCondition(
                    key="model_id",
                    match=models.MatchAny(any=model_ids),
                )
            )

        if mature_only:
            must_conditions.append(
                models.FieldCondition(
                    key="mature",
                    match=models.MatchValue(value=True),
                )
            )

        if max_lapses is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="lapses",
                    range=models.Range(lte=max_lapses),
                )
            )

        if min_reps is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="reps",
                    range=models.Range(gte=min_reps),
                )
            )

        query_filter = (
            models.Filter(must=must_conditions, must_not=must_not_conditions)
            if must_conditions or must_not_conditions
            else None
        )

        supports_sparse = await self._collection_supports_sparse()
        use_sparse = (
            supports_sparse
            and query_sparse_vector is not None
            and bool(query_sparse_vector.indices)
            and bool(query_sparse_vector.values)
        )
        candidate_limit = max(limit, prefetch_limit or limit * 3)

        if use_sparse:
            results = await client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=query_sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=["note_id"],
            )
        else:
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                with_payload=["note_id"],
            )

        return [(int(hit.payload["note_id"]), hit.score) for hit in results.points if hit.payload]

    async def find_similar_to_note(
        self,
        note_id: int,
        limit: int = 10,
        min_score: float = 0.9,
        deck_names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        """Find notes similar to a given note.

        Args:
            note_id: Note ID to find similar notes for.
            limit: Maximum results.
            min_score: Minimum similarity score.
            deck_names: Optional deck filter.
            tags: Optional tag filter.

        Returns:
            List of (note_id, score) tuples.
        """
        client = await self.get_client()

        # Get the vector for the note
        points = await client.retrieve(
            collection_name=self.collection_name,
            ids=[note_id],
            with_vectors=True,
        )

        if not points or not points[0].vector:
            return []

        vector_data = points[0].vector
        if isinstance(vector_data, dict):
            dense_vector = vector_data.get(DENSE_VECTOR_NAME)
            if dense_vector is None:
                dense_vector = vector_data.get("dense")
            if not isinstance(dense_vector, list):
                return []
            query_vector = cast("list[float]", dense_vector)
        else:
            query_vector = cast("list[float]", vector_data)

        # Build filter conditions
        must_conditions: list[models.Condition] = []

        if deck_names:
            must_conditions.append(
                models.FieldCondition(
                    key="deck_names",
                    match=models.MatchAny(any=deck_names),
                )
            )

        if tags:
            must_conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=tags),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Search for similar vectors
        results = await client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            using=DENSE_VECTOR_NAME,
            limit=limit + 1,  # +1 to exclude self
            score_threshold=min_score,
            with_payload=["note_id"],
        )

        # Filter out self and return
        return [
            (int(hit.payload["note_id"]), hit.score)
            for hit in results.points
            if hit.payload and int(hit.payload["note_id"]) != note_id
        ][:limit]

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
                "sparse_enabled": self._collection_has_sparse(info),
            }
        except Exception as e:
            logger.warning(
                "qdrant_collection_info_failed",
                collection=self.collection_name,
                error=str(e),
                error_type=type(e).__name__,
            )
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
        except Exception as e:
            logger.warning(
                "qdrant_health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
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
