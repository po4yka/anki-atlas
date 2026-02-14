"""Cross-encoder reranking for second-stage retrieval refinement."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    """Protocol for candidate reranking implementations."""

    async def rerank(
        self,
        query: str,
        documents: list[tuple[int, str]],
    ) -> list[tuple[int, float]]:
        """Score candidate documents for a query and return (id, score) pairs."""
        ...


class CrossEncoderReranker:
    """Sentence-transformers CrossEncoder-based reranker."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    def _get_model(self) -> Any:
        """Lazily initialize the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: uv sync --extra embeddings-local"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run synchronous prediction for (query, text) pairs."""
        if not pairs:
            return []

        model = self._get_model()
        raw_scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        return [float(score) for score in raw_scores]

    async def rerank(
        self,
        query: str,
        documents: list[tuple[int, str]],
    ) -> list[tuple[int, float]]:
        """Score candidate documents for query and return (id, score)."""
        if not documents:
            return []

        pairs = [(query, text) for _, text in documents]
        scores = await asyncio.to_thread(self._predict_scores, pairs)
        return [
            (doc_id, score) for (doc_id, _), score in zip(documents, scores, strict=True)
        ]
