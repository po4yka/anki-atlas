use std::path::{Path, PathBuf};

use indexer::embeddings::EmbeddingProvider;

use crate::services::EmbeddingFingerprint;

pub(super) const EMBEDDING_VECTOR_SCHEMA: &str = "multimodal_v1";

pub(super) fn current_embedding_fingerprint(
    embedding: &dyn EmbeddingProvider,
) -> EmbeddingFingerprint {
    EmbeddingFingerprint {
        model: embedding.model_name().to_string(),
        dimension: embedding.dimension(),
        vector_schema: EMBEDDING_VECTOR_SCHEMA.to_string(),
    }
}

pub(super) fn derive_media_root_from_collection_path(collection_path: &Path) -> Option<PathBuf> {
    let parent = collection_path.parent()?;
    Some(parent.join("collection.media"))
}
