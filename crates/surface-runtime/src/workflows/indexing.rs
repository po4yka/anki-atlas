use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use indexer::embeddings::{EmbeddingInput, EmbeddingPart, EmbeddingProvider, EmbeddingTask};
use indexer::qdrant::{NotePayload, SemanticSearchHit, SparseVector, VectorRepository};
use indexer::service::{
    ChunkForIndexing, IndexProgressCallback, IndexProgressEvent, IndexService, IndexStats,
    MultimodalNoteForIndexing, NoteForIndexing,
};
use regex::Regex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool};

use super::progress::{SurfaceOperation, SurfaceProgressSink, emit_progress, map_index_progress};
use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct IndexExecutionSummary {
    pub force_reindex: bool,
    pub stats: IndexStats,
}

#[derive(Debug, Clone, FromRow)]
struct NoteIndexRow {
    note_id: i64,
    model_id: i64,
    fields_json: serde_json::Value,
    raw_fields: Option<String>,
    normalized_text: String,
    tags: Vec<String>,
    deck_names: Vec<String>,
    mature: bool,
    lapses: i32,
    reps: i32,
    fail_rate: Option<f64>,
}

const EMBEDDING_VECTOR_SCHEMA: &str = "multimodal_v1";

use crate::services::EmbeddingFingerprint;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaAssetRef {
    source_field: Option<String>,
    asset_rel_path: String,
    mime_type: String,
    modality: String,
    preview_label: String,
}

fn current_embedding_fingerprint(embedding: &dyn EmbeddingProvider) -> EmbeddingFingerprint {
    EmbeddingFingerprint {
        model: embedding.model_name().to_string(),
        dimension: embedding.dimension(),
        vector_schema: EMBEDDING_VECTOR_SCHEMA.to_string(),
    }
}

fn short_preview_label(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "asset".to_string();
    }
    let preview = trimmed.chars().take(80).collect::<String>();
    preview.replace('\n', " ")
}

fn detect_media_type(rel_path: &str) -> Option<(&'static str, &'static str)> {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())?;

    match ext.as_str() {
        "png" => Some(("image", "image/png")),
        "jpg" | "jpeg" => Some(("image", "image/jpeg")),
        "gif" => Some(("image", "image/gif")),
        "webp" => Some(("image", "image/webp")),
        "mp3" => Some(("audio", "audio/mpeg")),
        "wav" => Some(("audio", "audio/wav")),
        "ogg" => Some(("audio", "audio/ogg")),
        "m4a" => Some(("audio", "audio/mp4")),
        "aac" => Some(("audio", "audio/aac")),
        "flac" => Some(("audio", "audio/flac")),
        "mp4" => Some(("video", "video/mp4")),
        "mov" => Some(("video", "video/quicktime")),
        "webm" => Some(("video", "video/webm")),
        "pdf" => Some(("document", "application/pdf")),
        _ => None,
    }
}

fn sanitize_relative_asset_path(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty()
        || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("data:")
        || trimmed.starts_with('/')
    {
        return None;
    }
    let candidate = trimmed.replace('\\', "/");
    if candidate.split('/').any(|part| part == "..") {
        return None;
    }
    Some(candidate)
}

fn capture_asset_refs(
    content: &str,
    source_field: Option<&str>,
    sink: &mut Vec<MediaAssetRef>,
    seen: &mut HashSet<(Option<String>, String)>,
) {
    let source_field = source_field.map(ToString::to_string);
    let patterns = [
        r#"(?i)<img[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<video[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<source[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<a[^>]+href=["']?([^"' >]+\.pdf(?:\?[^"' >]*)?)"#,
        r#"(?i)<embed[^>]+src=["']?([^"' >]+\.pdf(?:\?[^"' >]*)?)"#,
    ];

    for pattern in patterns {
        let Ok(regex) = Regex::new(pattern) else {
            continue;
        };
        for captures in regex.captures_iter(content) {
            let Some(path_match) = captures.get(1) else {
                continue;
            };
            let Some(asset_rel_path) = sanitize_relative_asset_path(path_match.as_str()) else {
                continue;
            };
            let Some((modality, mime_type)) = detect_media_type(&asset_rel_path) else {
                continue;
            };
            let key = (source_field.clone(), asset_rel_path.clone());
            if !seen.insert(key) {
                continue;
            }
            sink.push(MediaAssetRef {
                source_field: source_field.clone(),
                preview_label: Path::new(&asset_rel_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| asset_rel_path.clone()),
                asset_rel_path,
                mime_type: mime_type.to_string(),
                modality: modality.to_string(),
            });
        }
    }

    let Ok(sound_regex) = Regex::new(r#"\[sound:([^\]]+)\]"#) else {
        return;
    };
    for captures in sound_regex.captures_iter(content) {
        let Some(path_match) = captures.get(1) else {
            continue;
        };
        let Some(asset_rel_path) = sanitize_relative_asset_path(path_match.as_str()) else {
            continue;
        };
        let Some((modality, mime_type)) = detect_media_type(&asset_rel_path) else {
            continue;
        };
        let key = (source_field.clone(), asset_rel_path.clone());
        if !seen.insert(key) {
            continue;
        }
        sink.push(MediaAssetRef {
            source_field: source_field.clone(),
            preview_label: Path::new(&asset_rel_path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| asset_rel_path.clone()),
            asset_rel_path,
            mime_type: mime_type.to_string(),
            modality: modality.to_string(),
        });
    }
}

fn extract_media_refs(
    fields_json: &serde_json::Value,
    raw_fields: Option<&str>,
) -> Vec<MediaAssetRef> {
    let mut refs = Vec::new();
    let mut seen = HashSet::new();

    if let Some(object) = fields_json.as_object() {
        for (field_name, value) in object {
            if let Some(content) = value.as_str() {
                capture_asset_refs(content, Some(field_name), &mut refs, &mut seen);
            }
        }
    }

    if let Some(raw_fields) = raw_fields {
        for raw_field in raw_fields.split('\u{1f}') {
            capture_asset_refs(raw_field, None, &mut refs, &mut seen);
        }
    }

    refs
}

fn derive_media_root_from_collection_path(collection_path: &Path) -> Option<PathBuf> {
    let parent = collection_path.parent()?;
    Some(parent.join("collection.media"))
}

pub struct IndexingService {
    db: PgPool,
    embedding: Arc<dyn EmbeddingProvider>,
    vector_repo: Arc<dyn VectorRepository>,
    anki_collection_path: Option<PathBuf>,
    anki_media_root: Option<PathBuf>,
}

#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait IndexExecutor: Send + Sync {
    async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError>;

    async fn index_all_notes_with_progress(
        &self,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError>;
}

impl IndexingService {
    pub fn unsupported(db: PgPool) -> Self {
        Self {
            db,
            embedding: Arc::new(indexer::embeddings::MockEmbeddingProvider::new(1)),
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

    async fn load_sync_metadata_value(&self, key: &str) -> Result<Option<String>, SurfaceError> {
        sqlx::query_scalar::<_, String>("SELECT value #>> '{}' FROM sync_metadata WHERE key = $1")
            .bind(key)
            .fetch_optional(&self.db)
            .await
            .map_err(Into::into)
    }

    async fn store_sync_metadata_value(&self, key: &str, value: &str) -> Result<(), SurfaceError> {
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

    async fn load_stored_embedding_fingerprint(
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

    async fn store_embedding_fingerprint(&self) -> Result<(), SurfaceError> {
        let fingerprint = current_embedding_fingerprint(self.embedding.as_ref());
        self.store_sync_metadata_value("embedding_model", &fingerprint.model)
            .await?;
        self.store_sync_metadata_value("embedding_dimension", &fingerprint.dimension.to_string())
            .await?;
        self.store_sync_metadata_value("embedding_vector_schema", &fingerprint.vector_schema)
            .await?;
        Ok(())
    }

    async fn resolve_media_root(&self) -> Result<Option<PathBuf>, SurfaceError> {
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

    async fn prepare_collection(&self) -> Result<bool, SurfaceError> {
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

    fn build_chunk_id(note_id: i64, chunk_kind: &str, suffix: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(note_id.to_le_bytes());
        hasher.update(chunk_kind.as_bytes());
        hasher.update(suffix.as_bytes());
        format!(
            "{note_id}:{chunk_kind}:{}",
            hex::encode(&hasher.finalize()[..6])
        )
    }

    fn build_note_chunks(
        row: &NoteIndexRow,
        media_root: Option<&Path>,
    ) -> Result<MultimodalNoteForIndexing, SurfaceError> {
        let mut chunks = vec![ChunkForIndexing {
            chunk_id: format!("{}:text_primary", row.note_id),
            chunk_kind: "text_primary".to_string(),
            modality: "text".to_string(),
            embedding_input: EmbeddingInput::text_with_task(
                row.normalized_text.clone(),
                EmbeddingTask::RetrievalDocument,
            ),
            sparse_text: Some(row.normalized_text.clone()),
            source_field: None,
            asset_rel_path: None,
            mime_type: Some("text/plain".to_string()),
            preview_label: Some(short_preview_label(&row.normalized_text)),
            hash_component: row.normalized_text.clone(),
        }];

        if let Some(media_root) = media_root {
            for asset in extract_media_refs(&row.fields_json, row.raw_fields.as_deref()) {
                let asset_path = media_root.join(&asset.asset_rel_path);
                if !asset_path.exists() {
                    continue;
                }

                let bytes = std::fs::read(&asset_path)?;
                let digest = {
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    hex::encode(&hasher.finalize()[..8])
                };
                let suffix = format!(
                    "{}:{}",
                    asset.source_field.as_deref().unwrap_or("asset"),
                    asset.asset_rel_path
                );
                chunks.push(ChunkForIndexing {
                    chunk_id: Self::build_chunk_id(row.note_id, "asset", &suffix),
                    chunk_kind: "asset".to_string(),
                    modality: asset.modality.clone(),
                    embedding_input: EmbeddingInput {
                        parts: vec![EmbeddingPart::InlineBytes {
                            mime_type: asset.mime_type.clone(),
                            data: bytes,
                            display_name: Some(asset.preview_label.clone()),
                        }],
                        task: EmbeddingTask::RetrievalDocument,
                        title: Some(asset.preview_label.clone()),
                        output_dimensionality: None,
                    },
                    sparse_text: None,
                    source_field: asset.source_field.clone(),
                    asset_rel_path: Some(asset.asset_rel_path.clone()),
                    mime_type: Some(asset.mime_type.clone()),
                    preview_label: Some(asset.preview_label.clone()),
                    hash_component: digest,
                });
            }
        }

        Ok(MultimodalNoteForIndexing {
            note: NoteForIndexing {
                note_id: row.note_id,
                model_id: row.model_id,
                normalized_text: row.normalized_text.clone(),
                tags: row.tags.clone(),
                deck_names: row.deck_names.clone(),
                mature: row.mature,
                lapses: row.lapses,
                reps: row.reps,
                fail_rate: row.fail_rate,
            },
            chunks,
        })
    }

    pub async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        self.index_all_notes_with_progress(force_reindex, None)
            .await
    }

    pub async fn index_all_notes_with_progress(
        &self,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        let service = IndexService::new(self.embedding.clone(), self.vector_repo.clone());
        let force_reindex = force_reindex || self.prepare_collection().await?;
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
            .map(|row| Self::build_note_chunks(row, media_root.as_deref()))
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
            .index_multimodal_notes_with_progress(&notes, force_reindex, mapped_progress)
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
            force_reindex,
            stats,
        })
    }
}

#[async_trait::async_trait]
impl IndexExecutor for IndexingService {
    async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        Self::index_all_notes(self, force_reindex).await
    }

    async fn index_all_notes_with_progress(
        &self,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        Self::index_all_notes_with_progress(self, force_reindex, progress).await
    }
}

struct UnsupportedVectorRepository;

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
    ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
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
    ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
        Err(indexer::qdrant::VectorStoreError::Client(
            "vector repository is not configured".to_string(),
        ))
    }

    async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
        Ok(())
    }
}
