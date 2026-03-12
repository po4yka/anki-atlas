use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anki_sync::{
    SyncProgressCallback, SyncProgressEvent, SyncProgressStage, SyncStats,
    sync_anki_collection_owned_with_progress,
};
use generator::models::GeneratedCard;
use indexer::embeddings::EmbeddingProvider;
use indexer::qdrant::{NotePayload, SparseVector, VectorRepository};
use indexer::service::{
    IndexProgressCallback, IndexProgressEvent, IndexProgressStage, IndexService, IndexStats,
    NoteForIndexing,
};
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

pub type SurfaceProgressSink = Arc<dyn Fn(SurfaceProgressEvent) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SurfaceOperation {
    Sync,
    Index,
    ObsidianScan,
}

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceProgressEvent {
    pub operation: SurfaceOperation,
    pub stage: String,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

fn emit_progress(
    sink: Option<&SurfaceProgressSink>,
    operation: SurfaceOperation,
    stage: impl Into<String>,
    current: usize,
    total: usize,
    message: impl Into<String>,
) {
    if let Some(sink) = sink {
        sink(SurfaceProgressEvent {
            operation,
            stage: stage.into(),
            current,
            total,
            message: message.into(),
        });
    }
}

fn map_sync_progress(progress: &SyncProgressEvent) -> SurfaceProgressEvent {
    SurfaceProgressEvent {
        operation: SurfaceOperation::Sync,
        stage: progress.stage.as_str().to_string(),
        current: progress.current,
        total: progress.total,
        message: progress.message.clone(),
    }
}

fn map_index_progress(progress: &IndexProgressEvent) -> Option<SurfaceProgressEvent> {
    if progress.stage == IndexProgressStage::Completed {
        return None;
    }

    Some(SurfaceProgressEvent {
        operation: SurfaceOperation::Index,
        stage: progress.stage.as_str().to_string(),
        current: progress.current,
        total: progress.total,
        message: progress.message.clone(),
    })
}

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

impl Default for GeneratePreviewService {
    fn default() -> Self {
        Self::new()
    }
}

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

impl Default for ValidationService {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for ObsidianScanService {
    fn default() -> Self {
        Self::new()
    }
}

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
        self.scan_with_progress(vault, source_dirs, dry_run, None)
    }

    pub fn scan_with_progress(
        &self,
        vault: &Path,
        source_dirs: &[String],
        dry_run: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<ObsidianScanPreview, SurfaceError> {
        if !dry_run {
            return Err(SurfaceError::Unsupported(
                "obsidian persistence is not implemented; use --dry-run".to_string(),
            ));
        }
        if !vault.exists() {
            return Err(SurfaceError::PathNotFound(vault.to_path_buf()));
        }

        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "scanning_vault",
            0,
            1,
            format!("scanning vault {}", vault.display()),
        );
        let workflow = ObsidianSyncWorkflow::new(
            PreviewCardGenerator,
            progress.as_ref().map(|sink| {
                let sink = Arc::clone(sink);
                Box::new(move |phase: &str, current: usize, total: usize| {
                    sink(SurfaceProgressEvent {
                        operation: SurfaceOperation::ObsidianScan,
                        stage: phase.to_string(),
                        current,
                        total,
                        message: format!("{phase}: {current}/{total}"),
                    });
                }) as obsidian::sync::ProgressCallback
            }),
        );
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

        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "analyzing_vault",
            0,
            1,
            "analyzing vault structure",
        );
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

        let preview = ObsidianScanPreview {
            vault_path: vault.to_path_buf(),
            source_dirs: source_dirs.to_vec(),
            note_count: stats.total_notes,
            generated_cards: sync.generated,
            orphaned_notes: stats.orphaned_notes,
            broken_links: stats.broken_links,
            notes: note_previews,
        };
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "completed",
            preview.note_count,
            preview.note_count.max(1),
            format!(
                "obsidian scan completed with {} generated cards",
                preview.generated_cards
            ),
        );
        Ok(preview)
    }
}

pub struct TagAuditService;

impl Default for TagAuditService {
    fn default() -> Self {
        Self::new()
    }
}

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
        self.index_all_notes_with_progress(force_reindex, None)
            .await
    }

    pub async fn index_all_notes_with_progress(
        &self,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError> {
        let service = IndexService::new(self.embedding.clone(), self.vector_repo.clone());
        self.vector_repo
            .ensure_collection(self.embedding.dimension())
            .await?;

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
             GROUP BY n.note_id, n.model_id, n.normalized_text, n.tags
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
        let mapped_progress = progress.as_ref().map(|sink| {
            let sink = Arc::clone(sink);
            Arc::new(move |event: IndexProgressEvent| {
                if let Some(mapped) = map_index_progress(&event) {
                    sink(mapped);
                }
            }) as IndexProgressCallback
        });
        let mut stats = service
            .index_notes_with_progress(&notes, force_reindex, mapped_progress)
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

pub struct SyncExecutionService {
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
}

#[derive(Clone)]
pub struct SyncExecutionHandle {
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
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

    pub fn handle(&self) -> SyncExecutionHandle {
        SyncExecutionHandle {
            db: self.db.clone(),
            indexer: self.indexer.clone(),
        }
    }

    pub async fn sync_collection(
        &self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        self.sync_collection_with_progress(source, run_migrations, run_index, force_reindex, None)
            .await
    }

    pub async fn sync_collection_with_progress(
        &self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        run_sync_collection(
            self.db.clone(),
            self.indexer.clone(),
            source,
            run_migrations,
            run_index,
            force_reindex,
            progress,
        )
        .await
    }
}

impl SyncExecutionHandle {
    pub async fn sync_collection(
        self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        self.sync_collection_with_progress(source, run_migrations, run_index, force_reindex, None)
            .await
    }

    pub async fn sync_collection_with_progress(
        self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        run_sync_collection(
            self.db,
            self.indexer,
            source,
            run_migrations,
            run_index,
            force_reindex,
            progress,
        )
        .await
    }
}

async fn run_sync_collection(
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
    source: PathBuf,
    run_migrations: bool,
    run_index: bool,
    force_reindex: bool,
    progress: Option<SurfaceProgressSink>,
) -> Result<SyncExecutionSummary, SurfaceError> {
    if !source.exists() {
        return Err(SurfaceError::PathNotFound(source));
    }
    if run_migrations {
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Sync,
            "running_migrations",
            0,
            1,
            "running database migrations",
        );
        database::run_migrations_owned(db.clone()).await?;
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Sync,
            "running_migrations",
            1,
            1,
            "database migrations complete",
        );
    }
    let mapped_progress = progress.as_ref().map(|sink| {
        let sink = Arc::clone(sink);
        Arc::new(move |event: SyncProgressEvent| {
            sink(map_sync_progress(&event));
        }) as SyncProgressCallback
    });
    let sync =
        sync_anki_collection_owned_with_progress(db, source.clone(), mapped_progress).await?;
    let index = if run_index {
        Some(
            indexer
                .index_all_notes_with_progress(force_reindex, progress.clone())
                .await?,
        )
    } else {
        None
    };
    emit_progress(
        progress.as_ref(),
        SurfaceOperation::Sync,
        SyncProgressStage::Completed.as_str(),
        1,
        1,
        format!("sync pipeline completed for {}", source.display()),
    );

    Ok(SyncExecutionSummary {
        source,
        migrations_applied: run_migrations,
        sync: sync.into(),
        index,
    })
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

// ---- QdrantVectorStore ----

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

#[cfg(test)]
mod tests {
    use super::*;
    use anki_sync::SyncStats;
    use indexer::qdrant::SearchFilters;
    use obsidian::parser::ParsedNote;
    use obsidian::sync::CardGenerator;
    use qdrant_client::qdrant::point_id;
    use std::collections::HashMap;

    // --- build_filters tests ---

    fn dummy_store() -> QdrantVectorStore {
        let client = Qdrant::from_url("http://localhost:0").build().unwrap();
        QdrantVectorStore::new(client, "test")
    }

    #[test]
    fn build_filters_all_none_returns_none() {
        let store = dummy_store();
        let filters = SearchFilters::default();
        assert!(store.build_filters(&filters).is_none());
    }

    #[test]
    fn build_filters_deck_names_only() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec!["Deck1".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
        assert!(filter.must_not.is_empty());
    }

    #[test]
    fn build_filters_tags_exclude_only() {
        let store = dummy_store();
        let filters = SearchFilters {
            tags_exclude: Some(vec!["old".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert!(filter.must.is_empty());
        assert_eq!(filter.must_not.len(), 1);
    }

    #[test]
    fn build_filters_mature_only_flag() {
        let store = dummy_store();
        let filters = SearchFilters {
            mature_only: true,
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_min_reps_range() {
        let store = dummy_store();
        let filters = SearchFilters {
            min_reps: Some(5),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_max_lapses_range() {
        let store = dummy_store();
        let filters = SearchFilters {
            max_lapses: Some(3),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_combined_must_and_must_not() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec!["A".into()]),
            tags_exclude: Some(vec!["B".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
        assert_eq!(filter.must_not.len(), 1);
    }

    #[test]
    fn build_filters_empty_vecs_treated_as_none() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec![]),
            tags: Some(vec![]),
            tags_exclude: Some(vec![]),
            deck_names_exclude: Some(vec![]),
            model_ids: Some(vec![]),
            ..Default::default()
        };
        assert!(store.build_filters(&filters).is_none());
    }

    // --- parse_validation_input tests ---

    #[test]
    fn parse_validation_input_valid_three_parts() {
        let content = "What is Rust?\n---\nA systems language.\n---\ncs::rust\nprogramming";
        let (front, back, tags) = parse_validation_input(content).unwrap();
        assert_eq!(front, "What is Rust?");
        assert_eq!(back, "A systems language.");
        assert_eq!(tags, vec!["cs::rust", "programming"]);
    }

    #[test]
    fn parse_validation_input_two_parts_no_tags() {
        let content = "Front\n---\nBack";
        let (front, back, tags) = parse_validation_input(content).unwrap();
        assert_eq!(front, "Front");
        assert_eq!(back, "Back");
        assert!(tags.is_empty());
    }

    #[test]
    fn parse_validation_input_missing_back_returns_error() {
        let content = "Only front content";
        let result = parse_validation_input(content);
        assert!(matches!(result, Err(SurfaceError::InvalidInput(_))));
    }

    #[test]
    fn parse_validation_input_empty_front_returns_error() {
        let content = "\n---\nBack content";
        let result = parse_validation_input(content);
        assert!(matches!(result, Err(SurfaceError::InvalidInput(_))));
    }

    #[test]
    fn parse_validation_input_tags_trimmed_and_filtered() {
        let content = "Front\n---\nBack\n---\n  cs::basics  \n\n  rust  \n  ";
        let (_, _, tags) = parse_validation_input(content).unwrap();
        assert_eq!(tags, vec!["cs::basics", "rust"]);
    }

    // --- PreviewCardGenerator tests ---

    fn make_parsed_note(
        title: Option<&str>,
        sections: Vec<(&str, &str)>,
        body: &str,
    ) -> ParsedNote {
        ParsedNote {
            path: PathBuf::from("test.md"),
            frontmatter: HashMap::new(),
            content: String::new(),
            body: body.to_string(),
            sections: sections
                .into_iter()
                .map(|(h, c)| (h.to_string(), c.to_string()))
                .collect(),
            title: title.map(ToString::to_string),
        }
    }

    #[test]
    fn preview_card_generator_with_sections() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(
            Some("Test Note"),
            vec![("Heading 1", "Content 1"), ("Heading 2", "Content 2")],
            "",
        );
        let cards = generator.generate(&note);
        assert_eq!(cards.len(), 2);
        assert!(cards[0].apf_html.contains("Heading 1"));
        assert!(cards[1].apf_html.contains("Content 2"));
    }

    #[test]
    fn preview_card_generator_no_sections_generates_one() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(Some("Empty"), vec![], "Full body text");
        let cards = generator.generate(&note);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].apf_html, "Full body text");
    }

    #[test]
    fn preview_card_generator_slug_format() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(Some("My Great Note"), vec![("S1", "C1"), ("S2", "C2")], "");
        let cards = generator.generate(&note);
        assert_eq!(cards[0].slug, "my-great-note-1");
        assert_eq!(cards[1].slug, "my-great-note-2");
    }

    // --- SyncStatsSummary::from ---

    #[test]
    fn sync_stats_summary_from_maps_all_fields() {
        let stats = SyncStats {
            decks_upserted: 1,
            models_upserted: 2,
            notes_upserted: 3,
            notes_deleted: 4,
            cards_upserted: 5,
            card_stats_upserted: 6,
            duration_ms: 7,
        };
        let summary = SyncStatsSummary::from(stats);
        assert_eq!(summary.decks_upserted, 1);
        assert_eq!(summary.models_upserted, 2);
        assert_eq!(summary.notes_upserted, 3);
        assert_eq!(summary.notes_deleted, 4);
        assert_eq!(summary.cards_upserted, 5);
        assert_eq!(summary.card_stats_upserted, 6);
        assert_eq!(summary.duration_ms, 7);
    }

    // --- Default impls ---

    #[test]
    fn generate_preview_service_default_does_not_panic() {
        let _service: GeneratePreviewService = Default::default();
    }

    #[test]
    fn validation_service_default_does_not_panic() {
        let _service = ValidationService::default();
    }

    #[test]
    fn obsidian_scan_service_default_does_not_panic() {
        let _service: ObsidianScanService = Default::default();
    }

    #[test]
    fn tag_audit_service_default_does_not_panic() {
        let _service: TagAuditService = Default::default();
    }

    // --- note_id_from_point ---

    #[test]
    fn note_id_from_point_numeric_ok() {
        let store = dummy_store();
        let point = Some(PointId {
            point_id_options: Some(point_id::PointIdOptions::Num(42)),
        });
        assert_eq!(store.note_id_from_point(point).unwrap(), 42);
    }

    #[test]
    fn note_id_from_point_uuid_err() {
        let store = dummy_store();
        let point = Some(PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(
                "550e8400-e29b-41d4-a716-446655440000".into(),
            )),
        });
        assert!(store.note_id_from_point(point).is_err());
    }

    #[test]
    fn note_id_from_point_none_err() {
        let store = dummy_store();
        assert!(store.note_id_from_point(None).is_err());
    }

    // --- Send + Sync ---

    #[test]
    fn all_public_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeneratePreview>();
        assert_send_sync::<ValidationSummary>();
        assert_send_sync::<ObsidianNotePreview>();
        assert_send_sync::<ObsidianScanPreview>();
        assert_send_sync::<TagAuditEntry>();
        assert_send_sync::<TagAuditSummary>();
        assert_send_sync::<SyncExecutionSummary>();
        assert_send_sync::<IndexExecutionSummary>();
        assert_send_sync::<SyncStatsSummary>();
        assert_send_sync::<GeneratePreviewService>();
        assert_send_sync::<ValidationService>();
        assert_send_sync::<ObsidianScanService>();
        assert_send_sync::<TagAuditService>();
        assert_send_sync::<QdrantVectorStore>();
    }
}
