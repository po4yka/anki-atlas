"""RAG Service -- high-level API for retrieval-augmented generation.

Provides:
- Context enrichment during card generation
- Duplicate detection via semantic similarity
- Few-shot example retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from packages.common.logging import get_logger

if TYPE_CHECKING:
    from packages.rag.store import SearchResult, VaultVectorStore

logger = get_logger(module=__name__)


@dataclass(frozen=True, slots=True)
class RelatedConcept:
    """A related concept from the knowledge base."""

    title: str
    content: str
    topic: str
    similarity: float
    source_file: str


@dataclass(frozen=True, slots=True)
class DuplicateCheckResult:
    """Result of a duplicate detection check."""

    is_duplicate: bool
    confidence: float
    similar_items: tuple[SearchResult, ...]
    recommendation: str


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """A few-shot example for card generation."""

    question: str
    answer: str
    topic: str
    difficulty: str
    source_file: str


def _duplicate_recommendation(confidence: float) -> str:
    if confidence >= 0.95:
        return "Highly likely duplicate -- skip this card"
    if confidence >= 0.85:
        return "Probable duplicate -- review before creating"
    if confidence >= 0.70:
        return "Similar content exists -- consider differentiating"
    return "No significant duplicates found"


class RAGService:
    """High-level RAG operations for flashcard generation.

    All search methods accept pre-computed embedding vectors so the
    caller controls the embedding model.  This keeps ``RAGService``
    free of embedding-provider dependencies.
    """

    def __init__(self, store: VaultVectorStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_duplicates(
        self,
        query_embedding: list[float],
        *,
        threshold: float = 0.85,
        k: int = 5,
    ) -> DuplicateCheckResult:
        """Check whether a card is a potential duplicate.

        Args:
            query_embedding: Embedding of the card content.
            threshold: Similarity threshold for considering a duplicate.
            k: Maximum candidates to inspect.

        Returns:
            DuplicateCheckResult with confidence and similar items.
        """
        results = self._store.search(query_embedding, k=k, min_similarity=threshold)
        confidence = max((r.similarity for r in results), default=0.0)
        return DuplicateCheckResult(
            is_duplicate=len(results) > 0,
            confidence=confidence,
            similar_items=tuple(results),
            recommendation=_duplicate_recommendation(confidence),
        )

    def get_context(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        topic: str | None = None,
        min_similarity: float = 0.3,
    ) -> list[RelatedConcept]:
        """Retrieve related concepts for context enrichment.

        Args:
            query_embedding: Embedding of the source content.
            k: Number of results.
            topic: Optional topic filter.
            min_similarity: Minimum similarity threshold.

        Returns:
            De-duplicated list of RelatedConcept.
        """
        where: dict[str, Any] | None = None
        if topic:
            where = {"topic": {"$eq": topic}}

        results = self._store.search(
            query_embedding,
            k=k * 2,
            where=where,
            min_similarity=min_similarity,
        )

        seen: set[str] = set()
        concepts: list[RelatedConcept] = []
        for r in results:
            if r.source_file in seen:
                continue
            seen.add(r.source_file)
            concepts.append(
                RelatedConcept(
                    title=r.metadata.get("title", ""),
                    content=r.content,
                    topic=r.metadata.get("topic", ""),
                    similarity=r.similarity,
                    source_file=r.source_file,
                )
            )
            if len(concepts) >= k:
                break

        logger.debug("context_retrieved", results=len(concepts))
        return concepts

    def get_few_shot_examples(
        self,
        query_embedding: list[float],
        *,
        k: int = 3,
        topic: str | None = None,
    ) -> list[FewShotExample]:
        """Retrieve few-shot examples for generation prompts.

        Args:
            query_embedding: Embedding for the target topic/content.
            k: Number of examples.
            topic: Optional topic filter.

        Returns:
            List of FewShotExample.
        """
        where: dict[str, Any] | None = None
        if topic:
            where = {"topic": {"$eq": topic}}

        results = self._store.search(
            query_embedding,
            k=k * 3,
            where=where,
            min_similarity=0.3,
        )

        examples: list[FewShotExample] = []
        for r in results:
            examples.append(
                FewShotExample(
                    question=r.content[:300],
                    answer=r.content[:500],
                    topic=r.metadata.get("topic", ""),
                    difficulty=r.metadata.get("difficulty", "medium"),
                    source_file=r.source_file,
                )
            )
            if len(examples) >= k:
                break

        logger.debug("few_shot_examples_retrieved", count=len(examples))
        return examples
