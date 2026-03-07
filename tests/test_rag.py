"""Tests for packages.rag: chunker, store, and service."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from packages.rag.chunker import (
    ChunkType,
    DocumentChunk,
    DocumentChunker,
    _compute_content_hash,
    _generate_chunk_id,
)
from packages.rag.service import (
    RAGService,
    _duplicate_recommendation,
)
from packages.rag.store import SearchResult

# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------


class TestDocumentChunk:
    def test_frozen(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c1",
            content="hello",
            chunk_type=ChunkType.SECTION,
            source_file="a.md",
            content_hash=_compute_content_hash("hello"),
        )
        with pytest.raises(AttributeError):
            chunk.content = "changed"  # type: ignore[misc]

    def test_content_hash_deterministic(self) -> None:
        h1 = _compute_content_hash("foo")
        h2 = _compute_content_hash("foo")
        assert h1 == h2
        assert len(h1) == 16

    def test_generate_chunk_id(self) -> None:
        cid = _generate_chunk_id("/vault/note.md", "summary")
        assert "note" in cid
        assert "summary" in cid


class TestDocumentChunker:
    def test_empty_content(self) -> None:
        chunker = DocumentChunker()
        chunks = chunker.chunk_content("", "empty.md")
        assert chunks == []

    def test_plain_text_single_section(self) -> None:
        chunker = DocumentChunker()
        chunks = chunker.chunk_content(
            "This is plain text without any headings." * 5,
            "plain.md",
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.SECTION

    def test_heading_sections(self) -> None:
        content = (
            "# Summary\n"
            "This is a summary section with enough content to pass the minimum.\n\n"
            "# Answer\n"
            "This is the answer section with enough content to pass the minimum.\n"
        )
        chunker = DocumentChunker(min_chunk_size=10)
        chunks = chunker.chunk_content(content, "note.md")

        types = {c.chunk_type for c in chunks}
        assert ChunkType.SUMMARY in types
        assert ChunkType.ANSWER in types

    def test_code_block_extraction(self) -> None:
        content = (
            "Some text before code.\n\n"
            "```python\n"
            "def hello():\n"
            '    print("hello world")\n'
            "    return True\n"
            "```\n"
        )
        chunker = DocumentChunker(min_chunk_size=10)
        chunks = chunker.chunk_content(content, "code.md")

        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE_EXAMPLE]
        assert len(code_chunks) == 1
        assert "python" in code_chunks[0].metadata.get("code_language", "")

    def test_code_blocks_disabled(self) -> None:
        content = "```python\nprint('hi')\n```\n"
        chunker = DocumentChunker(include_code_blocks=False, min_chunk_size=5)
        chunks = chunker.chunk_content(content, "code.md")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE_EXAMPLE]
        assert code_chunks == []

    def test_frontmatter_metadata(self) -> None:
        chunker = DocumentChunker(min_chunk_size=5)
        chunks = chunker.chunk_content(
            "Some body content here.",
            "note.md",
            frontmatter={"title": "Test Note", "topic": "Python"},
        )
        assert chunks
        assert chunks[0].metadata["title"] == "Test Note"
        assert chunks[0].metadata["topic"] == "Python"

    def test_truncation(self) -> None:
        long_text = "word " * 500
        chunker = DocumentChunker(chunk_size=100, min_chunk_size=5)
        chunks = chunker.chunk_content(long_text, "long.md")
        assert chunks
        assert chunks[0].content.endswith("...")
        assert len(chunks[0].content) <= 105  # 100 + "..."

    def test_chunk_file_missing(self, tmp_path: Path) -> None:
        chunker = DocumentChunker()
        chunks = chunker.chunk_file(tmp_path / "missing.md")
        assert chunks == []

    def test_chunk_file_success(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("# Title\nSome content for the section.\n")
        chunker = DocumentChunker(min_chunk_size=5)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1

    def test_min_chunk_size_filter(self) -> None:
        content = "# Heading\nab\n"
        chunker = DocumentChunker(min_chunk_size=100)
        chunks = chunker.chunk_content(content, "small.md")
        assert chunks == []

    def test_section_classification(self) -> None:
        chunker = DocumentChunker()
        assert chunker._classify_section("Summary (EN)") == ChunkType.SUMMARY
        assert chunker._classify_section("Key Points") == ChunkType.KEY_POINTS
        assert chunker._classify_section("Question") == ChunkType.QUESTION
        assert chunker._classify_section("Answer") == ChunkType.ANSWER
        assert chunker._classify_section("Other") == ChunkType.SECTION


# ---------------------------------------------------------------------------
# store (SearchResult only -- VaultVectorStore requires chromadb)
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_similarity_from_score(self) -> None:
        r = SearchResult(
            chunk_id="c1",
            content="hello",
            score=0.0,
            source_file="a.md",
        )
        assert r.similarity == pytest.approx(1.0)

    def test_similarity_decreases_with_distance(self) -> None:
        r1 = SearchResult(chunk_id="c1", content="a", score=0.0, source_file="a.md")
        r2 = SearchResult(chunk_id="c2", content="b", score=1.0, source_file="b.md")
        assert r1.similarity > r2.similarity


# ---------------------------------------------------------------------------
# service (mock vector store)
# ---------------------------------------------------------------------------


def _mock_store(
    search_results: list[SearchResult] | None = None,
) -> MagicMock:
    store = MagicMock()
    store.search.return_value = search_results or []
    store.count.return_value = 0
    return store


class TestRAGService:
    def test_find_duplicates_none(self) -> None:
        svc = RAGService(_mock_store())
        result = svc.find_duplicates([0.1, 0.2, 0.3])
        assert not result.is_duplicate
        assert result.confidence == 0.0
        assert "No significant" in result.recommendation

    def test_find_duplicates_found(self) -> None:
        sr = SearchResult(
            chunk_id="c1",
            content="dup",
            score=0.0,  # distance=0 => similarity=1.0
            source_file="a.md",
            metadata={"topic": "test"},
        )
        svc = RAGService(_mock_store([sr]))
        result = svc.find_duplicates([0.1, 0.2])
        assert result.is_duplicate
        assert result.confidence == pytest.approx(1.0)
        assert "skip" in result.recommendation.lower()

    def test_get_context_deduplication(self) -> None:
        results = [
            SearchResult(
                chunk_id="c1",
                content="a",
                score=0.1,
                source_file="same.md",
                metadata={"title": "T"},
            ),
            SearchResult(
                chunk_id="c2",
                content="b",
                score=0.2,
                source_file="same.md",
                metadata={"title": "T"},
            ),
        ]
        svc = RAGService(_mock_store(results))
        concepts = svc.get_context([0.1], k=5)
        # same source file should be de-duplicated
        assert len(concepts) == 1
        assert concepts[0].source_file == "same.md"

    def test_get_context_with_topic(self) -> None:
        store = _mock_store()
        svc = RAGService(store)
        svc.get_context([0.1], topic="python")
        # Verify where clause was passed
        call_kwargs = store.search.call_args
        assert call_kwargs.kwargs["where"] == {"topic": {"$eq": "python"}}

    def test_get_few_shot_examples(self) -> None:
        sr = SearchResult(
            chunk_id="c1",
            content="example content",
            score=0.1,
            source_file="ex.md",
            metadata={"topic": "math", "difficulty": "hard"},
        )
        svc = RAGService(_mock_store([sr]))
        examples = svc.get_few_shot_examples([0.1], k=2)
        assert len(examples) == 1
        assert examples[0].topic == "math"
        assert examples[0].difficulty == "hard"

    def test_get_few_shot_respects_k(self) -> None:
        results = [
            SearchResult(
                chunk_id=f"c{i}",
                content=f"content {i}",
                score=0.1,
                source_file=f"f{i}.md",
                metadata={},
            )
            for i in range(10)
        ]
        svc = RAGService(_mock_store(results))
        examples = svc.get_few_shot_examples([0.1], k=3)
        assert len(examples) == 3


class TestDuplicateRecommendation:
    def test_thresholds(self) -> None:
        assert "skip" in _duplicate_recommendation(0.96).lower()
        assert "review" in _duplicate_recommendation(0.90).lower()
        assert "differentiating" in _duplicate_recommendation(0.75).lower()
        assert "no significant" in _duplicate_recommendation(0.50).lower()


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_public_api(self) -> None:
        from packages.rag import (
            DocumentChunker,
            RAGService,
        )

        assert RAGService is not None
        assert DocumentChunker is not None
