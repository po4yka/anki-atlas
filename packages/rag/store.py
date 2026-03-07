"""Vector store for RAG system using ChromaDB (lazy import).

Provides persistent vector storage with metadata filtering and
similarity search for vault-level retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from packages.common.exceptions import VectorStoreError
from packages.common.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(module=__name__)

COLLECTION_NAME = "obsidian_vault"


def _get_chromadb() -> Any:
    """Lazy import for chromadb."""
    try:
        import chromadb

        return chromadb
    except ImportError as e:
        msg = "chromadb is required: pip install anki-atlas[rag]"
        raise VectorStoreError(msg) from e


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result from vector similarity search."""

    chunk_id: str
    content: str
    score: float
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        """Convert distance to similarity (0-1)."""
        return 1.0 / (1.0 + self.score)


class VaultVectorStore:
    """ChromaDB-backed vector store for vault content.

    Supports:
    - Adding and deleting chunks by ID
    - Similarity search with metadata filtering
    - Persistent on-disk storage
    """

    def __init__(
        self,
        persist_directory: Path,
        *,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        chromadb = _get_chromadb()
        from chromadb.config import Settings as ChromaSettings

        self.persist_directory = persist_directory
        persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_initialized",
            path=str(persist_directory),
            existing=self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add documents to the collection.

        Returns:
            Number of documents added.
        """
        if not ids:
            return 0

        existing = set(self._collection.get()["ids"])
        new_idx = [i for i, cid in enumerate(ids) if cid not in existing]

        if not new_idx:
            return 0

        self._collection.add(
            ids=[ids[i] for i in new_idx],
            documents=[documents[i] for i in new_idx],
            embeddings=[embeddings[i] for i in new_idx],
            metadatas=[metadatas[i] for i in new_idx] if metadatas else None,
        )
        logger.debug("chunks_added", count=len(new_idx))
        return len(new_idx)

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file."""
        results = self._collection.get(
            where={"source_file": {"$eq": source_file}},
        )
        if not results["ids"]:
            return 0

        self._collection.delete(ids=results["ids"])
        logger.info("chunks_deleted", source_file=source_file, count=len(results["ids"]))
        return len(results["ids"])

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_reset")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents by embedding vector.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results.
            where: Optional ChromaDB where clause.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            Sorted list of SearchResult.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 / (1.0 + distance)
                if similarity < min_similarity:
                    continue

                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                out.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        score=distance,
                        source_file=meta.get("source_file", ""),
                        metadata=meta,
                    )
                )

        logger.debug("search_completed", results=len(out))
        return out

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return int(self._collection.count())

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        total = self._collection.count()
        all_data = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        topics: set[str] = set()
        for meta in all_data.get("metadatas") or []:
            if meta:
                if "source_file" in meta:
                    sources.add(meta["source_file"])
                if "topic" in meta:
                    topics.add(meta["topic"])

        return {
            "total_chunks": total,
            "unique_files": len(sources),
            "topics": sorted(topics),
        }
