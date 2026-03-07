"""Document chunker for RAG system.

Parses markdown files into structured chunks suitable for
vector embedding and retrieval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from packages.common.logging import get_logger

logger = get_logger(module=__name__)


class ChunkType(StrEnum):
    """Type of document chunk."""

    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    CODE_EXAMPLE = "code_example"
    QUESTION = "question"
    ANSWER = "answer"
    FULL_CONTENT = "full_content"
    SECTION = "section"


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A chunk of document content with metadata."""

    chunk_id: str
    content: str
    chunk_type: ChunkType
    source_file: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_content_hash(content: str) -> str:
    """Compute a short SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _generate_chunk_id(source_file: str, section: str) -> str:
    """Generate unique chunk ID from file path and section name."""
    path_hash = hashlib.sha256(Path(source_file).as_posix().encode()).hexdigest()[:8]
    stem = Path(source_file).stem
    return f"{stem}_{section}_{path_hash}"


# Heading patterns for section extraction
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Code block pattern
_CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


class DocumentChunker:
    """Split markdown documents into typed chunks for embedding.

    Extracts:
    - Sections by heading
    - Code blocks (optional)
    - Full content as fallback
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        min_chunk_size: int = 50,
        include_code_blocks: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_code_blocks = include_code_blocks

    def chunk_content(
        self,
        content: str,
        source_file: str,
        *,
        frontmatter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Parse and chunk markdown content.

        Args:
            content: Markdown content (body without frontmatter delimiters).
            source_file: Source file identifier for metadata.
            frontmatter: Optional parsed frontmatter dict.

        Returns:
            List of document chunks.
        """
        base_meta = self._base_metadata(frontmatter or {}, source_file)
        chunks: list[DocumentChunk] = []

        # Extract heading-based sections
        sections = self._extract_sections(content)
        for section_name, section_body in sections:
            stripped = section_body.strip()
            if len(stripped) < self.min_chunk_size:
                continue

            chunk_type = self._classify_section(section_name)
            truncated = self._truncate(stripped)
            cid = _generate_chunk_id(source_file, section_name or "intro")

            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    content=truncated,
                    chunk_type=chunk_type,
                    source_file=source_file,
                    content_hash=_compute_content_hash(truncated),
                    metadata={**base_meta, "section": section_name},
                )
            )

        # Extract code blocks
        if self.include_code_blocks:
            for i, match in enumerate(_CODE_BLOCK_RE.finditer(content)):
                code = match.group(2).strip()
                if len(code) < self.min_chunk_size:
                    continue
                language = match.group(1) or "unknown"
                cid = _generate_chunk_id(source_file, f"code_{i}")
                truncated = self._truncate(code)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        content=truncated,
                        chunk_type=ChunkType.CODE_EXAMPLE,
                        source_file=source_file,
                        content_hash=_compute_content_hash(truncated),
                        metadata={**base_meta, "code_language": language},
                    )
                )

        # Fallback: full content if no structured chunks
        if not chunks and len(content.strip()) >= self.min_chunk_size:
            truncated = self._truncate(content.strip())
            chunks.append(
                DocumentChunk(
                    chunk_id=_generate_chunk_id(source_file, "full"),
                    content=truncated,
                    chunk_type=ChunkType.FULL_CONTENT,
                    source_file=source_file,
                    content_hash=_compute_content_hash(truncated),
                    metadata=base_meta,
                )
            )

        logger.debug("content_chunked", source=source_file, chunks=len(chunks))
        return chunks

    def chunk_file(self, file_path: Path) -> list[DocumentChunk]:
        """Read and chunk a single markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            List of document chunks (empty on read errors).
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("chunk_file_read_error", path=str(file_path), error=str(e))
            return []

        return self.chunk_content(content, str(file_path))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _base_metadata(frontmatter: dict[str, Any], source_file: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"source_file": source_file}
        for key in ("title", "topic", "difficulty", "tags"):
            if key in frontmatter:
                val = frontmatter[key]
                meta[key] = ",".join(str(v) for v in val) if isinstance(val, list) else str(val)
        return meta

    @staticmethod
    def _extract_sections(body: str) -> list[tuple[str, str]]:
        """Split body into (heading_text, content) pairs."""
        headings = list(_HEADING_RE.finditer(body))
        parts: list[tuple[str, str]] = []

        if not headings:
            stripped = body.strip()
            if stripped:
                return [("", stripped)]
            return []

        # Content before first heading
        pre = body[: headings[0].start()].strip()
        if pre:
            parts.append(("", pre))

        for i, m in enumerate(headings):
            heading_text = m.group(2).strip()
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            section_content = body[start:end].strip()
            parts.append((heading_text, section_content))

        return parts

    @staticmethod
    def _classify_section(heading: str) -> ChunkType:
        """Classify section by heading text."""
        lower = heading.lower()
        if "summary" in lower:
            return ChunkType.SUMMARY
        if "key point" in lower:
            return ChunkType.KEY_POINTS
        if "question" in lower:
            return ChunkType.QUESTION
        if "answer" in lower:
            return ChunkType.ANSWER
        return ChunkType.SECTION

    def _truncate(self, content: str) -> str:
        """Truncate content to chunk_size at a word boundary."""
        if len(content) <= self.chunk_size:
            return content
        truncated = content[: self.chunk_size]
        last_space = truncated.rfind(" ")
        if last_space > self.chunk_size // 2:
            truncated = truncated[:last_space]
        return truncated + "..."
