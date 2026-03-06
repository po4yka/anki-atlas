# Spec 16: RAG System

## Goal

Migrate the retrieval-augmented generation system into `packages/rag/`.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/rag/rag_service.py` -- `RAGService`, `get_rag_service()`, `DuplicateCheckResult`, `RelatedConcept`, `FewShotExample`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/rag/document_chunker.py` -- `DocumentChunker`, `DocumentChunk`, `ChunkType`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/rag/vector_store.py` -- `VaultVectorStore`, `SearchResult`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/rag/embedding_provider.py` -- `EmbeddingProvider`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/rag/integration.py` -- `RAGIntegration`

Note: anki-atlas already has `packages/indexer/` with embedding and Qdrant support. The RAG system here is separate -- it uses ChromaDB for local vault-level retrieval, not the global Qdrant index.

## Target

### `packages/rag/` directory:

- `service.py` -- `RAGService` (main service), `DuplicateCheckResult`, `RelatedConcept`
- `chunker.py` -- `DocumentChunker`, `DocumentChunk`, `ChunkType` enum
- `store.py` -- `VaultVectorStore` (ChromaDB-backed), `SearchResult`
- `__init__.py` -- Re-export key classes

### Key design decisions:

- ChromaDB is in `rag` extras group (lazy import)
- `packages/rag/` is for vault-level RAG (card generation context)
- `packages/indexer/` remains for global search index (Qdrant)
- These two systems serve different purposes and should not be merged

## Acceptance Criteria

- [ ] `packages/rag/` contains service.py, chunker.py, store.py
- [ ] `RAGService` provides: `find_duplicates()`, `get_context()`, `get_few_shot_examples()`
- [ ] `DocumentChunker` splits documents into typed chunks
- [ ] ChromaDB import is lazy (only when RAGService is instantiated)
- [ ] `from packages.rag import RAGService, DocumentChunker` works
- [ ] Tests in `tests/test_rag.py` cover: chunking, duplicate detection (mock vector store)
- [ ] `make check` passes
