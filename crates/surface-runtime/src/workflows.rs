use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anki_sync::SyncStats;
use generator::models::GeneratedCard;
use indexer::embeddings::EmbeddingProvider;
use indexer::qdrant::{NotePayload, SparseVector, VectorRepository};
use indexer::service::{IndexService, IndexStats, NoteForIndexing};
use obsidian::analyzer::VaultAnalyzer;
use obsidian::parser::{ParsedNote, parse_note};
use obsidian::sync::{CardGenerator, GeneratedCardRef, ObsidianSyncWorkflow};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, GetPointsBuilder,
    PointId, PointStruct, RecommendPointsBuilder, SearchPointsBuilder, VectorParamsBuilder,
    point_id,
};
use serde::Serialize;
use sqlx::{FromRow, PgPool};
use taxonomy::{normalize_tag, suggest_tag, validate_tag};
use validation::pipeline::{ValidationIssue, ValidationPipeline};
use validation::quality::QualityScore;
use validation::validators::{ContentValidator, FormatValidator, HtmlValidator, TagValidator};

use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct GeneratePreview {
    pub source_file: PathBuf,
    pub title: Option<String>,
    pub sections: Vec<String>,
    pub estimated_cards: usize,
    pub warnings: Vec<String>,
    pub cards: Vec<GeneratedCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub source_file: PathBuf,
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub quality: Option<QualityScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObsidianNotePreview {
    pub path: PathBuf,
    pub title: Option<String>,
    pub sections: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObsidianScanPreview {
    pub vault_path: PathBuf,
    pub source_dirs: Vec<String>,
    pub note_count: usize,
    pub generated_cards: usize,
    pub orphaned_notes: Vec<String>,
    pub broken_links: Vec<(String, String)>,
    pub notes: Vec<ObsidianNotePreview>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagAuditEntry {
    pub tag: String,
    pub valid: bool,
    pub normalized: String,
    pub suggestion: Option<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagAuditSummary {
    pub source_file: PathBuf,
    pub applied_fixes: bool,
    pub entries: Vec<TagAuditEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncExecutionSummary {
    pub source: PathBuf,
    pub migrations_applied: bool,
    pub sync: SyncStatsSummary,
    pub index: Option<IndexExecutionSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexExecutionSummary {
    pub force_reindex: bool,
    pub stats: IndexStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncStatsSummary {
    pub decks_upserted: i32,
    pub models_upserted: i32,
    pub notes_upserted: i32,
    pub notes_deleted: i32,
    pub cards_upserted: i32,
    pub card_stats_upserted: i32,
    pub duration_ms: i64,
}

impl From<SyncStats> for SyncStatsSummary {
    fn from(stats: SyncStats) -> Self {
        Self {
            decks_upserted: stats.decks_upserted,
            models_upserted: stats.models_upserted,
            notes_upserted: stats.notes_upserted,
            notes_deleted: stats.notes_deleted,
            cards_upserted: stats.cards_upserted,
            card_stats_upserted: stats.card_stats_upserted,
            duration_ms: stats.duration_ms,
        }
    }
}

#[derive(Default)]
struct PreviewCardGenerator;

impl CardGenerator for PreviewCardGenerator {
    fn generate(&self, note: &ParsedNote) -> Vec<GeneratedCardRef> {
        let sections: Vec<_> = note
            .sections
            .iter()
            .filter(|(heading, content)| !heading.trim().is_empty() || !content.trim().is_empty())
            .collect();
        let count = if sections.is_empty() {
            1
        } else {
            sections.len()
        };

        (0..count)
            .map(|idx| GeneratedCardRef {
                slug: format!(
                    "{}-{}",
                    note.title
                        .as_deref()
                        .unwrap_or("note")
                        .to_lowercase()
                        .replace(' ', "-"),
                    idx + 1
                ),
                apf_html: note
                    .sections
                    .get(idx)
                    .map(|(heading, content)| format!("<h2>{heading}</h2>\n<p>{content}</p>"))
                    .unwrap_or_else(|| note.body.clone()),
            })
            .collect()
    }
}

pub struct GeneratePreviewService;

impl GeneratePreviewService {
    pub fn new() -> Self {
        Self
    }

    pub fn preview(&self, file: &Path) -> Result<GeneratePreview, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let note = parse_note(file, file.parent())?;
        let estimated_cards = note
            .sections
            .iter()
            .filter(|(_, content)| !content.trim().is_empty())
            .count()
            .max(1);
        let warnings = if note.title.is_none() {
            vec!["No title detected; using filename in previews.".to_string()]
        } else {
            Vec::new()
        };
        let cards = note
            .sections
            .iter()
            .enumerate()
            .filter(|(_, (_, content))| !content.trim().is_empty())
            .map(|(idx, (heading, content))| GeneratedCard {
                card_index: (idx + 1) as u32,
                slug: format!(
                    "{}-{}",
                    note.title
                        .as_deref()
                        .unwrap_or("note")
                        .to_lowercase()
                        .replace(' ', "-"),
                    idx + 1
                ),
                lang: "unknown".to_string(),
                apf_html: format!("<h2>{heading}</h2>\n<p>{content}</p>"),
                confidence: 0.5,
                content_hash: indexer::embeddings::content_hash("preview", content),
            })
            .collect();

        Ok(GeneratePreview {
            source_file: file.to_path_buf(),
            title: note.title,
            sections: note.sections.into_iter().map(|(name, _)| name).collect(),
            estimated_cards,
            warnings,
            cards,
        })
    }
}

pub struct ValidationService {
    pipeline: ValidationPipeline,
}

impl ValidationService {
    pub fn new() -> Self {
        Self {
            pipeline: ValidationPipeline::new(vec![
                Box::new(ContentValidator::new()),
                Box::new(FormatValidator::new()),
                Box::new(HtmlValidator::new()),
                Box::new(TagValidator::new()),
            ]),
        }
    }

    pub fn validate_file(
        &self,
        file: &Path,
        include_quality: bool,
    ) -> Result<ValidationSummary, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let content = std::fs::read_to_string(file)?;
        let (front, back, tags) = parse_validation_input(&content)?;
        let result = self.pipeline.run(&front, &back, &tags);
        let quality = include_quality.then(|| validation::quality::assess_quality(&front, &back));

        Ok(ValidationSummary {
            source_file: file.to_path_buf(),
            is_valid: result.is_valid(),
            issues: result.issues,
            quality,
        })
    }
}

fn parse_validation_input(content: &str) -> Result<(String, String, Vec<String>), SurfaceError> {
    let mut parts = content.splitn(3, "\n---\n");
    let front = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            SurfaceError::InvalidInput(
                "validation input must contain front content before the first `---` separator"
                    .to_string(),
            )
        })?
        .to_string();
    let back = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            SurfaceError::InvalidInput(
                "validation input must contain back content after the first `---` separator"
                    .to_string(),
            )
        })?
        .to_string();
    let tags = parts
        .next()
        .map(|chunk| {
            chunk
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok((front, back, tags))
}

pub struct ObsidianScanService;

impl ObsidianScanService {
    pub fn new() -> Self {
        Self
    }

    pub fn scan(
        &self,
        vault: &Path,
        source_dirs: &[String],
        dry_run: bool,
    ) -> Result<ObsidianScanPreview, SurfaceError> {
        if !dry_run {
            return Err(SurfaceError::Unsupported(
                "obsidian persistence is not implemented; use --dry-run".to_string(),
            ));
        }
        if !vault.exists() {
            return Err(SurfaceError::PathNotFound(vault.to_path_buf()));
        }

        let workflow = ObsidianSyncWorkflow::new(PreviewCardGenerator, None);
        let dir_refs: Vec<&str> = source_dirs.iter().map(String::as_str).collect();
        let notes = if dir_refs.is_empty() {
            workflow.scan_vault(vault, None)?
        } else {
            workflow.scan_vault(vault, Some(&dir_refs))?
        };
        let sync = if dir_refs.is_empty() {
            workflow.run(vault, None)?
        } else {
            workflow.run(vault, Some(&dir_refs))?
        };

        let mut analyzer = VaultAnalyzer::new(vault);
        let stats = analyzer.analyze()?;
        let note_previews = notes
            .into_iter()
            .map(|note| ObsidianNotePreview {
                path: note.path,
                title: note.title,
                sections: note.sections.len(),
            })
            .collect();

        Ok(ObsidianScanPreview {
            vault_path: vault.to_path_buf(),
            source_dirs: source_dirs.to_vec(),
            note_count: stats.total_notes,
            generated_cards: sync.generated,
            orphaned_notes: stats.orphaned_notes,
            broken_links: stats.broken_links,
            notes: note_previews,
        })
    }
}

pub struct TagAuditService;

impl TagAuditService {
    pub fn new() -> Self {
        Self
    }

    pub fn audit_file(
        &self,
        file: &Path,
        apply_fixes: bool,
    ) -> Result<TagAuditSummary, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let content = std::fs::read_to_string(file)?;
        let original_tags: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();

        let entries: Vec<TagAuditEntry> = original_tags
            .iter()
            .map(|tag| {
                let normalized = normalize_tag(tag);
                let suggestion = suggest_tag(tag, 3).into_iter().next();
                let validation = validate_tag(tag);
                let mut issues = validation;
                if normalized != *tag {
                    issues.push(format!("normalized form would be `{normalized}`"));
                }
                TagAuditEntry {
                    tag: tag.clone(),
                    valid: issues.is_empty(),
                    normalized,
                    suggestion,
                    issues,
                }
            })
            .collect();

        if apply_fixes {
            let normalized_tags: BTreeSet<String> = entries
                .iter()
                .map(|entry| entry.normalized.clone())
                .collect();
            let rewritten = normalized_tags.into_iter().collect::<Vec<_>>().join("\n");
            std::fs::write(file, format!("{rewritten}\n"))?;
        }

        Ok(TagAuditSummary {
            source_file: file.to_path_buf(),
            applied_fixes: apply_fixes,
            entries,
        })
    }
}

#[derive(Debug, Clone, FromRow)]
struct NoteIndexRow {
    note_id: i64,
    model_id: i64,
    normalized_text: String,
    tags: Vec<String>,
    deck_names: Vec<String>,
    mature: bool,
    lapses: i32,
    reps: i32,
    fail_rate: Option<f64>,
}

pub struct IndexingService {
    db: PgPool,
    embedding: Arc<dyn EmbeddingProvider>,
    vector_repo: Arc<dyn VectorRepository>,
}

#[async_trait::async_trait]
pub trait IndexExecutor: Send + Sync {
    async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError>;
}

impl IndexingService {
    pub fn unsupported(db: PgPool) -> Self {
        Self {
            db,
            embedding: Arc::new(indexer::embeddings::MockEmbeddingProvider::new(1)),
            vector_repo: Arc::new(UnsupportedVectorRepository),
        }
    }

    pub fn new(
        db: PgPool,
        embedding: Arc<dyn EmbeddingProvider>,
        vector_repo: Arc<dyn VectorRepository>,
    ) -> Self {
        Self {
            db,
            embedding,
            vector_repo,
        }
    }

    pub async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        let service = IndexService::new(self.embedding.clone(), self.vector_repo.clone());
        self.vector_repo
            .ensure_collection(self.embedding.dimension())
            .await?;

        let active_rows = sqlx::query_as::<_, NoteIndexRow>(
            "SELECT n.note_id,
                    n.model_id,
                    n.normalized_text,
                    n.tags,
                    COALESCE(array_remove(array_agg(DISTINCT d.name), NULL), '{}') AS deck_names,
                    COALESCE(bool_or(c.ivl >= 21), false) AS mature,
                    COALESCE(max(c.lapses), 0) AS lapses,
                    COALESCE(max(c.reps), 0) AS reps,
                    MAX(cs.fail_rate) AS fail_rate
             FROM notes n
             LEFT JOIN cards c ON c.note_id = n.note_id
             LEFT JOIN decks d ON d.deck_id = c.deck_id
             LEFT JOIN card_stats cs ON cs.card_id = c.card_id
             WHERE n.deleted_at IS NULL
             GROUP BY n.note_id, n.model_id, n.normalized_text, n.tags
             ORDER BY n.note_id",
        )
        .fetch_all(&self.db)
        .await?;

        let notes: Vec<NoteForIndexing> = active_rows
            .into_iter()
            .map(|row| NoteForIndexing {
                note_id: row.note_id,
                model_id: row.model_id,
                normalized_text: row.normalized_text,
                tags: row.tags,
                deck_names: row.deck_names,
                mature: row.mature,
                lapses: row.lapses,
                reps: row.reps,
                fail_rate: row.fail_rate,
            })
            .collect();

        let deleted_note_ids: Vec<i64> =
            sqlx::query_scalar("SELECT note_id FROM notes WHERE deleted_at IS NOT NULL")
                .fetch_all(&self.db)
                .await?;
        let mut stats = service.index_notes(&notes, force_reindex).await?;
        if !deleted_note_ids.is_empty() {
            stats.notes_deleted = self.vector_repo.delete_vectors(&deleted_note_ids).await?;
        }

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
}

pub struct SyncExecutionService {
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
}

#[async_trait::async_trait]
pub trait SyncExecutor: Send + Sync {
    async fn sync_collection(
        &self,
        source: &Path,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
    ) -> Result<SyncExecutionSummary, SurfaceError>;
}

impl SyncExecutionService {
    pub fn unsupported(db: PgPool) -> Self {
        Self {
            indexer: Arc::new(IndexingService::unsupported(db.clone())),
            db,
        }
    }

    pub fn new(db: PgPool, indexer: Arc<dyn IndexExecutor>) -> Self {
        Self { db, indexer }
    }

    pub async fn sync_collection(
        &self,
        source: &Path,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        if !source.exists() {
            return Err(SurfaceError::PathNotFound(source.to_path_buf()));
        }
        if run_migrations {
            database::run_migrations(&self.db).await?;
        }
        let sync = anki_sync::sync_anki_collection(&self.db, source).await?;
        let index = if run_index {
            Some(self.indexer.index_all_notes(force_reindex).await?)
        } else {
            None
        };

        Ok(SyncExecutionSummary {
            source: source.to_path_buf(),
            migrations_applied: run_migrations,
            sync: sync.into(),
            index,
        })
    }
}

#[async_trait::async_trait]
impl SyncExecutor for SyncExecutionService {
    async fn sync_collection(
        &self,
        source: &Path,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        Self::sync_collection(self, source, run_migrations, run_index, force_reindex).await
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

pub struct QdrantVectorStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantVectorStore {
    pub fn new(client: Qdrant, collection_name: impl Into<String>) -> Self {
        Self {
            client,
            collection_name: collection_name.into(),
        }
    }

    fn build_filters(&self, filters: &indexer::qdrant::SearchFilters) -> Option<Filter> {
        let mut must = Vec::new();
        let mut must_not = Vec::new();

        if let Some(deck_names) = filters.deck_names.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("deck_names", deck_names));
        }
        if let Some(tags) = filters.tags.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("tags", tags));
        }
        if let Some(model_ids) = filters.model_ids.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("model_id", model_ids));
        }
        if filters.mature_only {
            must.push(Condition::matches("mature", true));
        }
        if let Some(min_reps) = filters.min_reps {
            must.push(Condition::range(
                "reps",
                qdrant_client::qdrant::Range {
                    gte: Some(f64::from(min_reps)),
                    ..Default::default()
                },
            ));
        }
        if let Some(max_lapses) = filters.max_lapses {
            must.push(Condition::range(
                "lapses",
                qdrant_client::qdrant::Range {
                    lte: Some(f64::from(max_lapses)),
                    ..Default::default()
                },
            ));
        }
        if let Some(deck_names_exclude) = filters
            .deck_names_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("deck_names", deck_names_exclude));
        }
        if let Some(tags_exclude) = filters
            .tags_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("tags", tags_exclude));
        }

        match (must.is_empty(), must_not.is_empty()) {
            (true, true) => None,
            (false, true) => Some(Filter::must(must)),
            (true, false) => Some(Filter::must_not(must_not)),
            (false, false) => Some(Filter {
                must,
                should: Vec::new(),
                must_not,
                min_should: None,
            }),
        }
    }

    fn note_id_from_point(
        &self,
        point_id: Option<qdrant_client::qdrant::PointId>,
    ) -> Result<i64, indexer::qdrant::VectorStoreError> {
        match point_id.and_then(|id| id.point_id_options) {
            Some(point_id::PointIdOptions::Num(value)) => i64::try_from(value).map_err(|_| {
                indexer::qdrant::VectorStoreError::Client(format!(
                    "point id {value} does not fit into i64"
                ))
            }),
            Some(point_id::PointIdOptions::Uuid(value)) => {
                Err(indexer::qdrant::VectorStoreError::Client(format!(
                    "uuid point ids are not supported for note-backed storage: {value}"
                )))
            }
            None => Err(indexer::qdrant::VectorStoreError::Client(
                "Qdrant point id missing".to_string(),
            )),
        }
    }
}

#[async_trait::async_trait]
impl VectorRepository for QdrantVectorStore {
    async fn ensure_collection(
        &self,
        dimension: usize,
    ) -> Result<bool, indexer::qdrant::VectorStoreError> {
        let exists = self
            .client
            .collection_exists(&self.collection_name)
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Connection(error.to_string()))?;
        if exists {
            return Ok(false);
        }

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimension as u64, Distance::Cosine)),
            )
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
        Ok(true)
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        _sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        if vectors.len() != payloads.len() {
            return Err(indexer::qdrant::VectorStoreError::Client(
                "vectors and payloads must have the same length".to_string(),
            ));
        }
        let points = vectors
            .iter()
            .zip(payloads.iter())
            .map(|(vector, payload)| {
                let json = serde_json::to_value(payload).map_err(|error| {
                    indexer::qdrant::VectorStoreError::Client(error.to_string())
                })?;
                let qdrant_payload = Payload::try_from(json).map_err(|error| {
                    indexer::qdrant::VectorStoreError::Client(error.to_string())
                })?;
                Ok(PointStruct::new(
                    payload.note_id as u64,
                    vector.clone(),
                    qdrant_payload,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.client
            .upsert_points(
                qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection_name, points)
                    .wait(true),
            )
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
        Ok(vectors.len())
    }

    async fn delete_vectors(
        &self,
        note_ids: &[i64],
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        if note_ids.is_empty() {
            return Ok(0);
        }
        let ids: Vec<PointId> = note_ids
            .iter()
            .map(|id| {
                u64::try_from(*id).map(|value| value.into()).map_err(|_| {
                    indexer::qdrant::VectorStoreError::Client(format!(
                        "note id {id} cannot be represented as a qdrant numeric id"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .wait(true)
                    .points(ids),
            )
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
        Ok(note_ids.len())
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, String>, indexer::qdrant::VectorStoreError> {
        if note_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let ids: Vec<PointId> = note_ids
            .iter()
            .map(|id| {
                u64::try_from(*id).map(|value| value.into()).map_err(|_| {
                    indexer::qdrant::VectorStoreError::Client(format!(
                        "note id {id} cannot be represented as a qdrant numeric id"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection_name, ids)
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;

        let mut hashes = std::collections::HashMap::new();
        for point in response.result {
            let note_id = self.note_id_from_point(point.id)?;
            let payload = Payload::from(point.payload);
            let payload: NotePayload = payload
                .deserialize()
                .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
            hashes.insert(note_id, payload.content_hash);
        }
        Ok(hashes)
    }

    async fn search(
        &self,
        query_vector: &[f32],
        _query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &indexer::qdrant::SearchFilters,
    ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
        let mut request =
            SearchPointsBuilder::new(&self.collection_name, query_vector.to_vec(), limit as u64);
        if let Some(filter) = self.build_filters(filters) {
            request = request.filter(filter);
        }

        let response = self
            .client
            .search_points(request)
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
        response
            .result
            .into_iter()
            .map(|point| Ok((self.note_id_from_point(point.id)?, point.score)))
            .collect()
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
        let filters = indexer::qdrant::SearchFilters {
            deck_names: deck_names.map(|items| items.to_vec()),
            tags: tags.map(|items| items.to_vec()),
            ..Default::default()
        };
        let mut request = RecommendPointsBuilder::new(&self.collection_name, (limit + 1) as u64)
            .add_positive(u64::try_from(note_id).map_err(|_| {
                indexer::qdrant::VectorStoreError::Client(format!(
                    "note id {note_id} cannot be represented as a qdrant numeric id"
                ))
            })?)
            .score_threshold(min_score);
        if let Some(filter) = self.build_filters(&filters) {
            request = request.filter(filter);
        }

        let response = self
            .client
            .recommend(request)
            .await
            .map_err(|error| indexer::qdrant::VectorStoreError::Client(error.to_string()))?;
        let mut results = Vec::new();
        for point in response.result {
            let found_note_id = self.note_id_from_point(point.id)?;
            if found_note_id != note_id {
                results.push((found_note_id, point.score));
            }
            if results.len() == limit {
                break;
            }
        }
        Ok(results)
    }

    async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
        Ok(())
    }
}
