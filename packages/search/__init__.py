"""Hybrid search module for Anki notes."""

from packages.search.fts import (
    FTSResult,
    LexicalSearchResult,
    SearchFilters,
    search_fts,
    search_lexical,
)
from packages.search.fusion import FusionStats, SearchResult, reciprocal_rank_fusion
from packages.search.reranker import CrossEncoderReranker, Reranker
from packages.search.service import (
    HybridSearchResult,
    NoteDetail,
    SearchService,
    hybrid_search,
)

__all__ = [
    "CrossEncoderReranker",
    "FTSResult",
    "FusionStats",
    "HybridSearchResult",
    "LexicalSearchResult",
    "NoteDetail",
    "Reranker",
    "SearchFilters",
    "SearchResult",
    "SearchService",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "search_fts",
    "search_lexical",
]
