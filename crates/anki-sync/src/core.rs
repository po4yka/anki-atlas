use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use sqlx::PgPool;

use anki_reader::normalizer::{build_card_deck_map, build_deck_map, normalize_notes};
use anki_reader::read_anki_collection;
use common::error::{AnkiAtlasError, Result};

/// Progress callback for long-running sync work.
pub type SyncProgressCallback = Arc<dyn Fn(SyncProgressEvent) + Send + Sync>;

/// Stage-level progress emitted during collection sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncProgressStage {
    Starting,
    ReadingCollection,
    NormalizingCollection,
    UpsertingDecks,
    UpsertingModels,
    UpsertingNotes,
    DeletingNotes,
    UpsertingCards,
    UpsertingCardStats,
    UpdatingMetadata,
    Completed,
}

impl SyncProgressStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::ReadingCollection => "reading_collection",
            Self::NormalizingCollection => "normalizing_collection",
            Self::UpsertingDecks => "upserting_decks",
            Self::UpsertingModels => "upserting_models",
            Self::UpsertingNotes => "upserting_notes",
            Self::DeletingNotes => "deleting_notes",
            Self::UpsertingCards => "upserting_cards",
            Self::UpsertingCardStats => "upserting_card_stats",
            Self::UpdatingMetadata => "updating_metadata",
            Self::Completed => "completed",
        }
    }
}

/// Progress event emitted during collection sync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncProgressEvent {
    pub stage: SyncProgressStage,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

fn emit_progress(
    callback: Option<&SyncProgressCallback>,
    stage: SyncProgressStage,
    current: usize,
    total: usize,
    message: impl Into<String>,
) {
    if let Some(callback) = callback {
        callback(SyncProgressEvent {
            stage,
            current,
            total,
            message: message.into(),
        });
    }
}

/// Convert any error into `AnkiAtlasError::Sync`.
fn sync_err(e: impl std::fmt::Display) -> AnkiAtlasError {
    AnkiAtlasError::Sync {
        message: e.to_string(),
        context: Default::default(),
    }
}

/// Statistics from a PostgreSQL sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub decks_upserted: i32,
    pub models_upserted: i32,
    pub notes_upserted: i32,
    pub notes_deleted: i32,
    pub cards_upserted: i32,
    pub card_stats_upserted: i32,
    pub duration_ms: i64,
}

/// High-level service for syncing an Anki collection to PostgreSQL.
pub struct SyncService {
    pool: PgPool,
}

impl SyncService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Read collection from SQLite, normalize notes, upsert all data to PostgreSQL.
    pub async fn sync_collection(&self, collection_path: impl AsRef<Path>) -> Result<SyncStats> {
        run_sync_collection(
            self.pool.clone(),
            collection_path.as_ref().to_path_buf(),
            None,
        )
        .await
    }

    /// Read collection from SQLite and emit progress while syncing.
    pub async fn sync_collection_with_progress(
        &self,
        collection_path: impl AsRef<Path>,
        progress: Option<SyncProgressCallback>,
    ) -> Result<SyncStats> {
        run_sync_collection(
            self.pool.clone(),
            collection_path.as_ref().to_path_buf(),
            progress,
        )
        .await
    }
}

async fn run_sync_collection(
    pool: PgPool,
    collection_path: PathBuf,
    progress: Option<SyncProgressCallback>,
) -> Result<SyncStats> {
    let start = Instant::now();

    emit_progress(
        progress.as_ref(),
        SyncProgressStage::Starting,
        0,
        1,
        format!("starting sync for {}", collection_path.display()),
    );

    // 1. Read collection
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::ReadingCollection,
        0,
        1,
        "reading anki collection",
    );
    let mut collection = read_anki_collection(&collection_path).map_err(sync_err)?;
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::ReadingCollection,
        1,
        1,
        "anki collection loaded",
    );

    // 2. Build maps and normalize notes
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::NormalizingCollection,
        0,
        1,
        "normalizing notes and deck relationships",
    );
    let deck_map = build_deck_map(&collection.decks);
    let card_deck_map = build_card_deck_map(&collection.cards);
    normalize_notes(&mut collection.notes, &deck_map, &card_deck_map);
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::NormalizingCollection,
        1,
        1,
        "collection normalized",
    );

    // 3. Build card_id set for filtering card_stats
    let card_ids: HashSet<i64> = collection.cards.iter().map(|c| c.card_id.0).collect();

    let mut stats = SyncStats::default();

    // 4. Single transaction for all upserts
    let mut tx = pool.begin().await.map_err(sync_err)?;

    // 4a. Upsert decks
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpsertingDecks,
        0,
        collection.decks.len(),
        "upserting decks",
    );
    for deck in &collection.decks {
        sqlx::query(
            "INSERT INTO decks (deck_id, name, parent_name, config)
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT (deck_id) DO UPDATE SET
                   name = EXCLUDED.name,
                   parent_name = EXCLUDED.parent_name,
                   config = EXCLUDED.config",
        )
        .bind(deck.deck_id.0)
        .bind(&deck.name)
        .bind(&deck.parent_name)
        .bind(&deck.config)
        .execute(&mut *tx)
        .await
        .map_err(sync_err)?;
        stats.decks_upserted += 1;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpsertingDecks,
            stats.decks_upserted as usize,
            collection.decks.len(),
            format!("upserted deck {}", deck.name),
        );
    }

    // 4b. Upsert models
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpsertingModels,
        0,
        collection.models.len(),
        "upserting models",
    );
    for model in &collection.models {
        let fields_json = serde_json::to_value(&model.fields).unwrap_or_default();
        let templates_json = serde_json::to_value(&model.templates).unwrap_or_default();

        sqlx::query(
            "INSERT INTO models (model_id, name, fields, templates, config)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (model_id) DO UPDATE SET
                   name = EXCLUDED.name,
                   fields = EXCLUDED.fields,
                   templates = EXCLUDED.templates,
                   config = EXCLUDED.config",
        )
        .bind(model.model_id.0)
        .bind(&model.name)
        .bind(&fields_json)
        .bind(&templates_json)
        .bind(&model.config)
        .execute(&mut *tx)
        .await
        .map_err(sync_err)?;
        stats.models_upserted += 1;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpsertingModels,
            stats.models_upserted as usize,
            collection.models.len(),
            format!("upserted model {}", model.name),
        );
    }

    // 4c. Upsert notes (set deleted_at = NULL)
    let note_ids: Vec<i64> = collection.notes.iter().map(|n| n.note_id.0).collect();
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpsertingNotes,
        0,
        collection.notes.len(),
        "upserting notes",
    );
    for note in &collection.notes {
        let fields_json = serde_json::to_value(&note.fields_json).unwrap_or_default();
        let tags: Vec<&str> = note.tags.iter().map(|s| s.as_str()).collect();

        sqlx::query(
                "INSERT INTO notes (note_id, model_id, tags, fields_json, raw_fields, normalized_text, mtime, usn, deleted_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULL)
                 ON CONFLICT (note_id) DO UPDATE SET
                   model_id = EXCLUDED.model_id,
                   tags = EXCLUDED.tags,
                   fields_json = EXCLUDED.fields_json,
                   raw_fields = EXCLUDED.raw_fields,
                   normalized_text = EXCLUDED.normalized_text,
                   mtime = EXCLUDED.mtime,
                   usn = EXCLUDED.usn,
                   deleted_at = NULL",
            )
            .bind(note.note_id.0)
            .bind(note.model_id.0)
            .bind(&tags)
            .bind(&fields_json)
            .bind(&note.raw_fields)
            .bind(&note.normalized_text)
            .bind(note.mtime)
            .bind(note.usn)
            .execute(&mut *tx)
            .await
            .map_err(sync_err)?;
        stats.notes_upserted += 1;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpsertingNotes,
            stats.notes_upserted as usize,
            collection.notes.len(),
            format!("upserted note {}", note.note_id),
        );
    }

    // 4d. Soft-delete notes not in collection
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::DeletingNotes,
        0,
        1,
        "soft-deleting missing notes",
    );
    let deleted = sqlx::query_scalar::<_, i64>(
        "UPDATE notes SET deleted_at = NOW()
             WHERE deleted_at IS NULL
               AND note_id != ALL($1)
             RETURNING note_id",
    )
    .bind(&note_ids)
    .fetch_all(&mut *tx)
    .await
    .map_err(sync_err)?;
    stats.notes_deleted = deleted.len() as i32;
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::DeletingNotes,
        1,
        1,
        format!("soft-deleted {} notes", stats.notes_deleted),
    );

    // 4e. Upsert cards
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpsertingCards,
        0,
        collection.cards.len(),
        "upserting cards",
    );
    for card in &collection.cards {
        sqlx::query(
                "INSERT INTO cards (card_id, note_id, deck_id, ord, due, ivl, ease, lapses, reps, queue, type, mtime, usn)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                 ON CONFLICT (card_id) DO UPDATE SET
                   note_id = EXCLUDED.note_id,
                   deck_id = EXCLUDED.deck_id,
                   ord = EXCLUDED.ord,
                   due = EXCLUDED.due,
                   ivl = EXCLUDED.ivl,
                   ease = EXCLUDED.ease,
                   lapses = EXCLUDED.lapses,
                   reps = EXCLUDED.reps,
                   queue = EXCLUDED.queue,
                   type = EXCLUDED.type,
                   mtime = EXCLUDED.mtime,
                   usn = EXCLUDED.usn",
            )
            .bind(card.card_id.0)
            .bind(card.note_id.0)
            .bind(card.deck_id.0)
            .bind(card.ord)
            .bind(card.due)
            .bind(card.ivl)
            .bind(card.ease)
            .bind(card.lapses)
            .bind(card.reps)
            .bind(card.queue)
            .bind(card.card_type)
            .bind(card.mtime)
            .bind(card.usn)
            .execute(&mut *tx)
            .await
            .map_err(sync_err)?;
        stats.cards_upserted += 1;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpsertingCards,
            stats.cards_upserted as usize,
            collection.cards.len(),
            format!("upserted card {}", card.card_id),
        );
    }

    // 4f. Upsert card_stats (skip orphaned)
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpsertingCardStats,
        0,
        collection.card_stats.len(),
        "upserting card stats",
    );
    let mut card_stats_processed = 0usize;
    for cs in &collection.card_stats {
        if !card_ids.contains(&cs.card_id.0) {
            card_stats_processed += 1;
            emit_progress(
                progress.as_ref(),
                SyncProgressStage::UpsertingCardStats,
                card_stats_processed,
                collection.card_stats.len(),
                format!("skipped orphaned card stats for {}", cs.card_id),
            );
            continue;
        }

        sqlx::query(
                "INSERT INTO card_stats (card_id, reviews, avg_ease, fail_rate, last_review_at, total_time_ms)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (card_id) DO UPDATE SET
                   reviews = EXCLUDED.reviews,
                   avg_ease = EXCLUDED.avg_ease,
                   fail_rate = EXCLUDED.fail_rate,
                   last_review_at = EXCLUDED.last_review_at,
                   total_time_ms = EXCLUDED.total_time_ms",
            )
            .bind(cs.card_id.0)
            .bind(cs.reviews)
            .bind(cs.avg_ease)
            .bind(cs.fail_rate)
            .bind(cs.last_review_at)
            .bind(cs.total_time_ms)
            .execute(&mut *tx)
            .await
            .map_err(sync_err)?;
        stats.card_stats_upserted += 1;
        card_stats_processed += 1;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpsertingCardStats,
            card_stats_processed,
            collection.card_stats.len(),
            format!("upserted card stats for {}", cs.card_id),
        );
    }

    // 4g. Update sync_metadata
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::UpdatingMetadata,
        0,
        2,
        "updating sync metadata",
    );
    let now = chrono::Utc::now().to_rfc3339();
    let collection_path_str = collection_path.display().to_string();

    for (idx, (key, value)) in [
        ("last_sync_at", now),
        ("last_collection_path", collection_path_str),
    ]
    .into_iter()
    .enumerate()
    {
        sqlx::query(
            "INSERT INTO sync_metadata (key, value) VALUES ($1, $2)
                 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        )
        .bind(key)
        .bind(serde_json::Value::String(value))
        .execute(&mut *tx)
        .await
        .map_err(sync_err)?;
        emit_progress(
            progress.as_ref(),
            SyncProgressStage::UpdatingMetadata,
            idx + 1,
            2,
            format!("updated sync metadata key `{key}`"),
        );
    }

    // Commit transaction
    tx.commit().await.map_err(sync_err)?;

    stats.duration_ms = start.elapsed().as_millis() as i64;
    emit_progress(
        progress.as_ref(),
        SyncProgressStage::Completed,
        1,
        1,
        format!("sync completed in {}ms", stats.duration_ms),
    );
    Ok(stats)
}

/// Convenience function.
pub async fn sync_anki_collection(
    pool: &PgPool,
    collection_path: impl AsRef<Path>,
) -> Result<SyncStats> {
    sync_anki_collection_owned(pool.clone(), collection_path.as_ref().to_path_buf()).await
}

pub async fn sync_anki_collection_owned(
    pool: PgPool,
    collection_path: PathBuf,
) -> Result<SyncStats> {
    run_sync_collection(pool, collection_path, None).await
}

pub async fn sync_anki_collection_owned_with_progress(
    pool: PgPool,
    collection_path: PathBuf,
    progress: Option<SyncProgressCallback>,
) -> Result<SyncStats> {
    run_sync_collection(pool, collection_path, progress).await
}
