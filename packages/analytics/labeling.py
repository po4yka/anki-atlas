"""Note-topic labeling using embedding similarity."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from psycopg import AsyncConnection

from packages.analytics.taxonomy import Taxonomy
from packages.common.config import Settings, get_settings
from packages.common.database import get_connection
from packages.indexer.embeddings import EmbeddingProvider, get_embedding_provider
from packages.indexer.qdrant import QdrantRepository, get_qdrant_repository


@dataclass
class TopicAssignment:
    """A topic assignment for a note."""

    note_id: int
    topic_id: int
    topic_path: str
    confidence: float
    method: str = "embedding"


@dataclass
class LabelingStats:
    """Statistics from a labeling operation."""

    notes_processed: int = 0
    assignments_created: int = 0
    topics_matched: int = 0


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TopicLabeler:
    """Service for labeling notes with topics."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        """Initialize labeler.

        Args:
            settings: Application settings.
            embedding_provider: Embedding provider.
            qdrant_repository: Qdrant repository.
        """
        self.settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._qdrant_repository = qdrant_repository

    async def get_embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    async def get_qdrant_repository(self) -> QdrantRepository:
        """Get or create Qdrant repository."""
        if self._qdrant_repository is None:
            self._qdrant_repository = await get_qdrant_repository(self.settings)
        return self._qdrant_repository

    async def embed_topics(
        self,
        taxonomy: Taxonomy,
    ) -> dict[str, list[float]]:
        """Embed all topic descriptions.

        Args:
            taxonomy: Taxonomy with topics to embed.

        Returns:
            Dictionary mapping topic path to embedding vector.
        """
        provider = await self.get_embedding_provider()

        # Build texts to embed (use description or label)
        topic_texts: list[tuple[str, str]] = []
        for topic in taxonomy.all_topics():
            text = topic.description or topic.label
            topic_texts.append((topic.path, text))

        if not topic_texts:
            return {}

        # Embed all texts
        texts = [t[1] for t in topic_texts]
        vectors = await provider.embed(texts)

        return {topic_texts[i][0]: vectors[i] for i in range(len(topic_texts))}

    async def label_notes(
        self,
        taxonomy: Taxonomy,
        min_confidence: float = 0.3,
        max_topics_per_note: int = 3,
        batch_size: int = 100,
    ) -> LabelingStats:
        """Label all notes with matching topics.

        Args:
            taxonomy: Taxonomy to use for labeling.
            min_confidence: Minimum similarity threshold.
            max_topics_per_note: Maximum topics to assign per note.
            batch_size: Batch size for processing.

        Returns:
            Labeling statistics.
        """
        stats = LabelingStats()

        # Embed topics
        topic_embeddings = await self.embed_topics(taxonomy)
        if not topic_embeddings:
            return stats

        stats.topics_matched = len(topic_embeddings)

        # Get all note embeddings from database
        async with get_connection(self.settings) as conn:
            # Count notes
            result = await conn.execute(
                "SELECT COUNT(*) as count FROM notes WHERE deleted_at IS NULL"
            )
            row = await result.fetchone()
            total_notes = row["count"] if row else 0

            # Process in batches
            offset = 0
            while offset < total_notes:
                assignments = await self._label_batch(
                    conn,
                    taxonomy,
                    topic_embeddings,
                    offset,
                    batch_size,
                    min_confidence,
                    max_topics_per_note,
                )

                stats.notes_processed += batch_size
                stats.assignments_created += len(assignments)

                # Store assignments
                if assignments:
                    await self._store_assignments(conn, assignments)

                offset += batch_size

            await conn.commit()

        return stats

    async def _label_batch(
        self,
        conn: AsyncConnection[dict[str, Any]],
        taxonomy: Taxonomy,
        topic_embeddings: dict[str, list[float]],
        offset: int,
        limit: int,
        min_confidence: float,
        max_topics_per_note: int,
    ) -> list[TopicAssignment]:
        """Label a batch of notes."""
        provider = await self.get_embedding_provider()

        # Fetch notes
        result = await conn.execute(
            """
            SELECT note_id, normalized_text
            FROM notes
            WHERE deleted_at IS NULL
            ORDER BY note_id
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            {"limit": limit, "offset": offset},
        )

        notes: list[tuple[int, str]] = []
        async for row in result:
            notes.append((row["note_id"], row["normalized_text"]))

        if not notes:
            return []

        # Embed note texts
        note_texts = [n[1] for n in notes]
        note_vectors = await provider.embed(note_texts)

        # Find best topics for each note
        assignments: list[TopicAssignment] = []
        topic_paths = list(topic_embeddings.keys())
        topic_vectors = [topic_embeddings[p] for p in topic_paths]

        for i, (note_id, _) in enumerate(notes):
            note_vec = note_vectors[i]

            # Compute similarities to all topics
            similarities: list[tuple[str, float]] = []
            for j, topic_path in enumerate(topic_paths):
                sim = cosine_similarity(note_vec, topic_vectors[j])
                if sim >= min_confidence:
                    similarities.append((topic_path, sim))

            # Sort by similarity and take top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:max_topics_per_note]

            for topic_path, confidence in top_matches:
                topic = taxonomy.get(topic_path)
                if topic and topic.topic_id:
                    assignments.append(
                        TopicAssignment(
                            note_id=note_id,
                            topic_id=topic.topic_id,
                            topic_path=topic_path,
                            confidence=confidence,
                        )
                    )

        return assignments

    async def _store_assignments(
        self,
        conn: AsyncConnection[dict[str, Any]],
        assignments: list[TopicAssignment],
    ) -> None:
        """Store topic assignments in database."""
        for assignment in assignments:
            await conn.execute(
                """
                INSERT INTO note_topics (note_id, topic_id, confidence, method)
                VALUES (%(note_id)s, %(topic_id)s, %(confidence)s, %(method)s)
                ON CONFLICT (note_id, topic_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method
                """,
                {
                    "note_id": assignment.note_id,
                    "topic_id": assignment.topic_id,
                    "confidence": assignment.confidence,
                    "method": assignment.method,
                },
            )

    async def label_single_note(
        self,
        note_id: int,
        taxonomy: Taxonomy,
        topic_embeddings: dict[str, list[float]] | None = None,
        min_confidence: float = 0.3,
        max_topics: int = 3,
    ) -> list[TopicAssignment]:
        """Label a single note with topics.

        Args:
            note_id: Note ID to label.
            taxonomy: Taxonomy to use.
            topic_embeddings: Pre-computed topic embeddings (optional).
            min_confidence: Minimum similarity threshold.
            max_topics: Maximum topics to assign.

        Returns:
            List of topic assignments.
        """
        provider = await self.get_embedding_provider()

        # Get topic embeddings if not provided
        if topic_embeddings is None:
            topic_embeddings = await self.embed_topics(taxonomy)

        if not topic_embeddings:
            return []

        # Get note text
        async with get_connection(self.settings) as conn:
            result = await conn.execute(
                "SELECT normalized_text FROM notes WHERE note_id = %(note_id)s",
                {"note_id": note_id},
            )
            row = await result.fetchone()
            if not row:
                return []

            note_text = row["normalized_text"]

        # Embed note
        note_vector = await provider.embed_single(note_text)

        # Find matching topics
        assignments: list[TopicAssignment] = []
        for topic_path, topic_vec in topic_embeddings.items():
            sim = cosine_similarity(note_vector, topic_vec)
            if sim >= min_confidence:
                topic = taxonomy.get(topic_path)
                if topic and topic.topic_id:
                    assignments.append(
                        TopicAssignment(
                            note_id=note_id,
                            topic_id=topic.topic_id,
                            topic_path=topic_path,
                            confidence=sim,
                        )
                    )

        # Sort and limit
        assignments.sort(key=lambda x: x.confidence, reverse=True)
        return assignments[:max_topics]


async def label_all_notes(
    taxonomy: Taxonomy,
    settings: Settings | None = None,
    min_confidence: float = 0.3,
) -> LabelingStats:
    """Convenience function to label all notes.

    Args:
        taxonomy: Taxonomy to use.
        settings: Application settings.
        min_confidence: Minimum confidence threshold.

    Returns:
        Labeling statistics.
    """
    labeler = TopicLabeler(settings)
    return await labeler.label_notes(taxonomy, min_confidence=min_confidence)
