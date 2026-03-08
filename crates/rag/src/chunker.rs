use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

/// Type of document chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "snake_case")]
pub enum ChunkType {
    Summary,
    KeyPoints,
    CodeExample,
    Question,
    Answer,
    FullContent,
    Section,
}

/// A chunk of document content with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub content: String,
    pub chunk_type: ChunkType,
    pub source_file: String,
    pub content_hash: String,
    pub metadata: HashMap<String, String>,
}

/// Configuration for the document chunker.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub min_chunk_size: usize,
    pub include_code_blocks: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            min_chunk_size: 50,
            include_code_blocks: true,
        }
    }
}

/// Split markdown documents into typed chunks for embedding.
pub struct DocumentChunker {
    config: ChunkerConfig,
}

impl DocumentChunker {
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Parse and chunk markdown content.
    pub fn chunk_content(
        &self,
        _content: &str,
        _source_file: &str,
        _frontmatter: Option<&HashMap<String, String>>,
    ) -> Vec<DocumentChunk> {
        // TODO: implement
        Vec::new()
    }

    /// Read and chunk a single markdown file.
    pub fn chunk_file(&self, _path: &std::path::Path) -> Vec<DocumentChunk> {
        // TODO: implement
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn default_chunker() -> DocumentChunker {
        DocumentChunker::new(ChunkerConfig::default())
    }

    // --- ChunkType strum serialization ---

    #[test]
    fn chunk_type_display_snake_case() {
        assert_eq!(ChunkType::KeyPoints.to_string(), "key_points");
        assert_eq!(ChunkType::CodeExample.to_string(), "code_example");
        assert_eq!(ChunkType::FullContent.to_string(), "full_content");
        assert_eq!(ChunkType::Summary.to_string(), "summary");
    }

    #[test]
    fn chunk_type_parse_from_snake_case() {
        assert_eq!("key_points".parse::<ChunkType>().unwrap(), ChunkType::KeyPoints);
        assert_eq!("summary".parse::<ChunkType>().unwrap(), ChunkType::Summary);
    }

    // --- ChunkerConfig defaults ---

    #[test]
    fn chunker_config_defaults() {
        let cfg = ChunkerConfig::default();
        assert_eq!(cfg.chunk_size, 1000);
        assert_eq!(cfg.min_chunk_size, 50);
        assert!(cfg.include_code_blocks);
    }

    // --- Heading-based section splitting ---

    #[test]
    fn chunk_content_splits_by_heading() {
        let chunker = default_chunker();
        let md = "# Introduction\n\nThis is the intro section with enough content to pass the minimum chunk size threshold easily.\n\n## Details\n\nHere are the details with sufficient content to meet the minimum chunk size requirement for testing.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks from 2 headings, got {}",
            chunks.len()
        );

        let types: Vec<ChunkType> = chunks.iter().map(|c| c.chunk_type).collect();
        assert!(
            types.contains(&ChunkType::Section),
            "Expected Section chunk type in {:?}",
            types
        );
    }

    #[test]
    fn chunk_content_classifies_summary_heading() {
        let chunker = default_chunker();
        let md = "# Summary\n\nThis is a summary of the document with enough text to pass min chunk size threshold easily.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty(), "Expected at least one chunk");
        let has_summary = chunks.iter().any(|c| c.chunk_type == ChunkType::Summary);
        assert!(has_summary, "Expected Summary type, got: {:?}", chunks);
    }

    #[test]
    fn chunk_content_classifies_key_points_heading() {
        let chunker = default_chunker();
        let md = "# Key Points\n\nThese are the key points of the document with enough content for chunk size.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty());
        let has_kp = chunks.iter().any(|c| c.chunk_type == ChunkType::KeyPoints);
        assert!(has_kp, "Expected KeyPoints type, got: {:?}", chunks);
    }

    #[test]
    fn chunk_content_classifies_question_heading() {
        let chunker = default_chunker();
        let md = "# Question\n\nWhat is the meaning of life? This needs to be long enough to pass the minimum chunk size.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty());
        let has_q = chunks.iter().any(|c| c.chunk_type == ChunkType::Question);
        assert!(has_q, "Expected Question type, got: {:?}", chunks);
    }

    #[test]
    fn chunk_content_classifies_answer_heading() {
        let chunker = default_chunker();
        let md = "# Answer\n\nThe answer to the question is 42, and here is more text to pass the minimum chunk size.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty());
        let has_a = chunks.iter().any(|c| c.chunk_type == ChunkType::Answer);
        assert!(has_a, "Expected Answer type, got: {:?}", chunks);
    }

    #[test]
    fn section_classification_is_case_insensitive() {
        let chunker = default_chunker();
        let md = "# SUMMARY OF FINDINGS\n\nThis is a summary with enough content to pass the minimum chunk size for testing.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty());
        let has_summary = chunks.iter().any(|c| c.chunk_type == ChunkType::Summary);
        assert!(has_summary, "Expected Summary for case-insensitive match");
    }

    // --- Code block extraction ---

    #[test]
    fn chunk_content_extracts_code_blocks() {
        let chunker = default_chunker();
        let md = "# Example\n\nSome introduction text that is long enough to meet the minimum chunk size requirement.\n\n```python\ndef hello():\n    print('hello world')\n    return True\n```\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        let code_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::CodeExample)
            .collect();
        assert!(
            !code_chunks.is_empty(),
            "Expected CodeExample chunks, got: {:?}",
            chunks
        );
    }

    // --- Fallback to FullContent ---

    #[test]
    fn chunk_content_falls_back_to_full_content() {
        let chunker = default_chunker();
        // No headings, no code blocks, just plain text
        let md = "This is just plain text without any markdown headings or code blocks but it is long enough to pass minimum.";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks.is_empty(), "Expected at least one chunk for plain text");
        let has_full = chunks.iter().any(|c| c.chunk_type == ChunkType::FullContent);
        assert!(has_full, "Expected FullContent fallback, got: {:?}", chunks);
    }

    // --- Min chunk size filtering ---

    #[test]
    fn chunk_content_skips_chunks_below_min_size() {
        let chunker = default_chunker(); // min_chunk_size = 50
        let md = "# Tiny\n\nHi\n\n# Big Section\n\nThis section has plenty of content to exceed the minimum chunk size threshold easily.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", None);

        // "Hi" (2 chars) should be skipped, "Big Section" content should remain
        for chunk in &chunks {
            assert!(
                chunk.content.len() >= 50 || chunk.chunk_type == ChunkType::CodeExample,
                "Chunk below min_chunk_size: len={}, content={:?}",
                chunk.content.len(),
                chunk.content
            );
        }
    }

    // --- Truncation at word boundary ---

    #[test]
    fn chunk_content_truncates_at_word_boundary() {
        let config = ChunkerConfig {
            chunk_size: 100,
            min_chunk_size: 10,
            ..ChunkerConfig::default()
        };
        let chunker = DocumentChunker::new(config);
        let long_text = format!(
            "# Section\n\n{}",
            "word ".repeat(50) // 250 chars, exceeds chunk_size of 100
        );
        let chunks = chunker.chunk_content(&long_text, "notes/test.md", None);

        assert!(!chunks.is_empty(), "Expected chunks from long content");
        for chunk in &chunks {
            if chunk.chunk_type == ChunkType::Section {
                assert!(
                    chunk.content.len() <= 103, // 100 + "..."
                    "Chunk exceeds max size: len={}",
                    chunk.content.len()
                );
                if chunk.content.len() > 50 {
                    assert!(
                        chunk.content.ends_with("..."),
                        "Truncated chunk should end with '...': {:?}",
                        chunk.content
                    );
                }
            }
        }
    }

    // --- Deterministic chunk IDs ---

    #[test]
    fn chunk_ids_are_deterministic() {
        let chunker = default_chunker();
        let md = "# Test Section\n\nThis is a test section with enough content to pass the minimum chunk size threshold.\n";
        let chunks1 = chunker.chunk_content(md, "notes/test.md", None);
        let chunks2 = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks1.is_empty());
        assert_eq!(chunks1.len(), chunks2.len());
        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(c1.chunk_id, c2.chunk_id, "Chunk IDs should be deterministic");
        }
    }

    #[test]
    fn chunk_id_contains_file_stem_and_section() {
        let chunker = default_chunker();
        let md = "# Introduction\n\nEnough content here to pass the minimum chunk size threshold for the test to work properly.\n";
        let chunks = chunker.chunk_content(md, "notes/my-file.md", None);

        assert!(!chunks.is_empty());
        let id = &chunks[0].chunk_id;
        assert!(
            id.contains("my-file"),
            "Chunk ID should contain file stem: {id}"
        );
    }

    // --- Deterministic content hashes ---

    #[test]
    fn content_hash_is_deterministic_sha256_prefix() {
        let chunker = default_chunker();
        let md = "# Test\n\nSome content that is long enough for the chunk to pass minimum size threshold easily.\n";
        let chunks1 = chunker.chunk_content(md, "notes/test.md", None);
        let chunks2 = chunker.chunk_content(md, "notes/test.md", None);

        assert!(!chunks1.is_empty());
        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(c1.content_hash, c2.content_hash);
            assert_eq!(c1.content_hash.len(), 16, "Content hash should be 16 hex chars");
            assert!(
                c1.content_hash.chars().all(|c| c.is_ascii_hexdigit()),
                "Content hash should be hex: {}",
                c1.content_hash
            );
        }
    }

    // --- Source file propagation ---

    #[test]
    fn chunks_have_correct_source_file() {
        let chunker = default_chunker();
        let md = "# Section\n\nEnough content to pass the minimum chunk size threshold for source file propagation test.\n";
        let chunks = chunker.chunk_content(md, "vault/notes/rust.md", None);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(chunk.source_file, "vault/notes/rust.md");
        }
    }

    // --- Frontmatter metadata ---

    #[test]
    fn chunk_content_includes_frontmatter_metadata() {
        let chunker = default_chunker();
        let mut fm = HashMap::new();
        fm.insert("topic".to_string(), "rust".to_string());
        fm.insert("tags".to_string(), "programming".to_string());

        let md = "# Section\n\nContent with frontmatter metadata that is long enough to pass minimum chunk size threshold.\n";
        let chunks = chunker.chunk_content(md, "notes/test.md", Some(&fm));

        assert!(!chunks.is_empty());
        // Frontmatter should appear in chunk metadata
        let chunk = &chunks[0];
        assert!(
            chunk.metadata.contains_key("topic") || chunk.metadata.contains_key("tags"),
            "Expected frontmatter in metadata: {:?}",
            chunk.metadata
        );
    }

    // --- chunk_file reads from disk ---

    #[test]
    fn chunk_file_reads_and_chunks_markdown() {
        let chunker = default_chunker();
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.md");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            writeln!(
                f,
                "# Hello\n\nThis is a test markdown file with enough content to pass the minimum chunk size threshold."
            )
            .unwrap();
        }

        let chunks = chunker.chunk_file(&file_path);
        assert!(!chunks.is_empty(), "Expected chunks from file");
    }

    #[test]
    fn chunk_file_nonexistent_returns_empty() {
        let chunker = default_chunker();
        let chunks = chunker.chunk_file(std::path::Path::new("/nonexistent/file.md"));
        assert!(chunks.is_empty(), "Expected empty for nonexistent file");
    }

    // --- Send + Sync compile-time assertions ---

    #[test]
    fn types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DocumentChunk>();
        assert_send_sync::<DocumentChunker>();
        assert_send_sync::<ChunkerConfig>();
        assert_send_sync::<ChunkType>();
    }
}
