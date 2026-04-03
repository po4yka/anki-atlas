use std::path::PathBuf;
use std::sync::Arc;

use common::ReindexMode;
use indexer::embeddings::EmbeddingProvider;
use indexer::qdrant::{NotePayload, SemanticSearchHit, SparseVector, VectorRepository};
use indexer::service::{
    IndexProgressCallback, IndexProgressEvent, IndexService, MultimodalNoteForIndexing,
};
use sqlx::{FromRow, PgPool};

use super::chunk::build_note_chunks;
use super::prepare::{current_embedding_fingerprint, derive_media_root_from_collection_path};
use super::{IndexExecutionSummary, IndexExecutor};
use crate::error::SurfaceError;
use crate::services::EmbeddingFingerprint;
use crate::workflows::progress::{
    SurfaceOperation, SurfaceProgressSink, emit_progress, map_index_progress,
};

#[derive(Debug, Clone, FromRow)]
pub(super) struct NoteIndexRow {
    pub(super) note_id: i64,
    pub(super) model_id: i64,
    pub(super) fields_json: serde_json::Value,
    pub(super) raw_fields: Option<String>,
    pub(super) normalized_text: String,
    pub(super) tags: Vec<String>,
    pub(super) deck_names: Vec<String>,
    pub(super) mature: bool,
    pub(super) lapses: i32,
    pub(super) reps: i32,
    pub(super) fail_rate: Option<f64>,
}

pub struct IndexingService {
    pub(super) db: PgPool,
    pub(super) embedding: Arc<dyn EmbeddingProvider>,
    pub(super) vector_repo: Arc<dyn VectorRepository>,
    pub(super) anki_collection_path: Option<PathBuf>,
    pub(super) anki_media_root: Option<PathBuf>,
}

impl IndexingService {
    pub fn unsupported(db: PgPool) -> Self {
        Self {
            db,
            embedding: Arc::new(indexer::embeddings::DeterministicEmbeddingProvider::new(1)),
            vector_repo: Arc::new(UnsupportedVectorRepository),
            anki_collection_path: None,
            anki_media_root: None,
        }
    }

    pub fn new(
        db: PgPool,
        embedding: Arc<dyn EmbeddingProvider>,
        vector_repo: Arc<dyn VectorRepository>,
        anki_collection_path: Option<PathBuf>,
        anki_media_root: Option<PathBuf>,
    ) -> Self {
        Self {
            db,
            embedding,
            vector_repo,
            anki_collection_path,
            anki_media_root,
        }
    }

    pub(super) async fn load_sync_metadata_value(
        &self,
        key: &str,
    ) -> Result<Option<String>, SurfaceError> {
        sqlx::query_scalar::<_, String>("SELECT value #>> '{}' FROM sync_metadata WHERE key = $1")
            .bind(key)
            .fetch_optional(&self.db)
            .await
            .map_err(Into::into)
    }

    pub(super) async fn store_sync_metadata_value(
        &self,
        key: &str,
        value: &str,
    ) -> Result<(), SurfaceError> {
        sqlx::query(
            "INSERT INTO sync_metadata (key, value) VALUES ($1, to_jsonb($2::text))
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        )
        .bind(key)
        .bind(value)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    pub(super) async fn load_stored_embedding_fingerprint(
        &self,
    ) -> Result<Option<EmbeddingFingerprint>, SurfaceError> {
        let model = self.load_sync_metadata_value("embedding_model").await?;
        let dimension = self.load_sync_metadata_value("embedding_dimension").await?;
        let vector_schema = self
            .load_sync_metadata_value("embedding_vector_schema")
            .await?;

        let (Some(model), Some(dimension), Some(vector_schema)) = (model, dimension, vector_schema)
        else {
            return Ok(None);
        };

        let dimension = dimension.parse::<usize>().map_err(|error| {
            SurfaceError::InvalidInput(format!(
                "stored embedding_dimension is invalid: {dimension} ({error})"
            ))
        })?;

        Ok(Some(EmbeddingFingerprint {
            model,
            dimension,
            vector_schema,
        }))
    }

    pub(super) async fn store_embedding_fingerprint(&self) -> Result<(), SurfaceError> {
        let fingerprint = current_embedding_fingerprint(self.embedding.as_ref());
        self.store_sync_metadata_value("embedding_model", &fingerprint.model)
            .await?;
        self.store_sync_metadata_value("embedding_dimension", &fingerprint.dimension.to_string())
            .await?;
        self.store_sync_metadata_value("embedding_vector_schema", &fingerprint.vector_schema)
            .await?;
        Ok(())
    }

    pub(super) async fn resolve_media_root(&self) -> Result<Option<PathBuf>, SurfaceError> {
        if let Some(path) = &self.anki_media_root {
            return Ok(Some(path.clone()));
        }

        if let Some(collection_path) = self
            .load_sync_metadata_value("last_collection_path")
            .await?
            .map(PathBuf::from)
            && let Some(media_root) = derive_media_root_from_collection_path(&collection_path)
        {
            return Ok(Some(media_root));
        }

        if let Some(collection_path) = &self.anki_collection_path
            && let Some(media_root) = derive_media_root_from_collection_path(collection_path)
        {
            return Ok(Some(media_root));
        }

        Ok(None)
    }

    pub(super) async fn prepare_collection(&self) -> Result<bool, SurfaceError> {
        let desired = current_embedding_fingerprint(self.embedding.as_ref());
        let stored = self.load_stored_embedding_fingerprint().await?;
        let current_dimension = self.vector_repo.collection_dimension().await?;

        let fingerprint_mismatch = stored.as_ref().is_some_and(|stored| {
            stored.model != desired.model
                || stored.dimension != desired.dimension
                || stored.vector_schema != desired.vector_schema
        });
        let dimension_mismatch =
            current_dimension.is_some_and(|dimension| dimension != desired.dimension);

        if fingerprint_mismatch || dimension_mismatch {
            self.vector_repo
                .recreate_collection(desired.dimension)
                .await?;
            return Ok(true);
        }

        if current_dimension.is_none() {
            self.vector_repo
                .ensure_collection(desired.dimension)
                .await?;
            return Ok(true);
        }

        Ok(false)
    }

    pub async fn index_all_notes(
        &self,
        reindex_mode: ReindexMode,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        self.index_all_notes_with_progress(reindex_mode, None).await
    }

    pub async fn index_all_notes_with_progress(
        &self,
        reindex_mode: ReindexMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        let service = IndexService::new(self.embedding.clone(), self.vector_repo.clone());
        let reindex_mode = if self.prepare_collection().await? {
            ReindexMode::Force
        } else {
            reindex_mode
        };
        let media_root = self.resolve_media_root().await?;

        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Index,
            "loading_notes",
            0,
            1,
            "loading active notes from postgres",
        );
        let active_rows = sqlx::query_as::<_, NoteIndexRow>(
            "SELECT n.note_id,
                    n.model_id,
                    n.fields_json,
                    n.raw_fields,
                    n.normalized_text,
                    n.tags,
                    COALESCE(array_remove(array_agg(DISTINCT d.name), NULL), '{}') AS deck_names,
                    COALESCE(bool_or(c.ivl >= 21), false) AS mature,
                    COALESCE(max(c.lapses), 0) AS lapses,
                    COALESCE(max(c.reps), 0) AS reps,
                    MAX(cs.fail_rate)::float8 AS fail_rate
             FROM notes n
             LEFT JOIN cards c ON c.note_id = n.note_id
             LEFT JOIN decks d ON d.deck_id = c.deck_id
             LEFT JOIN card_stats cs ON cs.card_id = c.card_id
             WHERE n.deleted_at IS NULL
             GROUP BY n.note_id, n.model_id, n.fields_json, n.raw_fields, n.normalized_text, n.tags
             ORDER BY n.note_id",
        )
        .fetch_all(&self.db)
        .await?;
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Index,
            "loading_notes",
            1,
            1,
            format!("loaded {} active notes", active_rows.len()),
        );

        let notes: Vec<MultimodalNoteForIndexing> = active_rows
            .iter()
            .map(|row| build_note_chunks(row, media_root.as_deref()))
            .collect::<Result<_, _>>()?;

        let deleted_note_ids: Vec<i64> =
            sqlx::query_scalar("SELECT note_id FROM notes WHERE deleted_at IS NOT NULL")
                .fetch_all(&self.db)
                .await?;
        let mapped_progress = progress.as_ref().map(|sink| {
            let sink = Arc::clone(sink);
            Arc::new(move |event: IndexProgressEvent| {
                if let Some(mapped) = map_index_progress(&event) {
                    sink(mapped);
                }
            }) as IndexProgressCallback
        });
        let mut stats = service
            .index_multimodal_notes_with_progress(&notes, reindex_mode, mapped_progress)
            .await?;
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Index,
            "delete_cleanup",
            0,
            deleted_note_ids.len().max(1),
            format!(
                "cleaning up {} deleted note vectors",
                deleted_note_ids.len()
            ),
        );
        if !deleted_note_ids.is_empty() {
            stats.notes_deleted = self.vector_repo.delete_vectors(&deleted_note_ids).await?;
        }
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Index,
            "delete_cleanup",
            deleted_note_ids.len(),
            deleted_note_ids.len().max(1),
            format!("deleted {} note vectors", stats.notes_deleted),
        );
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Index,
            "completed",
            stats.notes_processed,
            stats.notes_processed.max(1),
            format!(
                "index complete: {} embedded, {} skipped, {} deleted",
                stats.notes_embedded, stats.notes_skipped, stats.notes_deleted
            ),
        );
        self.store_embedding_fingerprint().await?;

        Ok(IndexExecutionSummary {
            reindex_mode,
            stats,
        })
    }
}

#[async_trait::async_trait]
impl IndexExecutor for IndexingService {
    async fn index_all_notes(
        &self,
        reindex_mode: ReindexMode,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        Self::index_all_notes(self, reindex_mode).await
    }

    async fn index_all_notes_with_progress(
        &self,
        reindex_mode: ReindexMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        Self::index_all_notes_with_progress(self, reindex_mode, progress).await
    }
}

pub(super) struct UnsupportedVectorRepository;

#[async_trait::async_trait]
impl VectorRepository for UnsupportedVectorRepository {
    async fn ensure_collection(
        &self,
        _dimension: usize,
    ) -> Result<bool, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn collection_dimension(
        &self,
    ) -> Result<Option<usize>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn recreate_collection(
        &self,
        _dimension: usize,
    ) -> Result<(), indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn upsert_vectors(
        &self,
        _vectors: &[Vec<f32>],
        _payloads: &[NotePayload],
        _sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn delete_vectors(
        &self,
        _note_ids: &[i64],
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn get_existing_hashes(
        &self,
        _note_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, String>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn search_chunks(
        &self,
        _query_vector: &[f32],
        _query_sparse: Option<&SparseVector>,
        _limit: usize,
        _filters: &indexer::qdrant::SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn search(
        &self,
        _query_vector: &[f32],
        _query_sparse: Option<&SparseVector>,
        _limit: usize,
        _filters: &indexer::qdrant::SearchFilters,
    ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn find_similar_to_note(
        &self,
        _note_id: i64,
        _limit: usize,
        _min_score: f32,
        _deck_names: Option<&[String]>,
        _tags: Option<&[String]>,
    ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
        Ok(())
    }
}
