use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};

/// Schema version for migration tracking.
pub const SCHEMA_VERSION: u32 = 2;

/// Column list for card queries (must match row_to_card_entry field order).
const CARD_COLUMNS: &str = "slug, note_id, source_path, front, back, content_hash, metadata_hash, \
     language, tags, anki_note_id, created_at, updated_at, synced_at";

/// Column list for note queries (must match row_to_note_entry field order).
const NOTE_COLUMNS: &str = "note_id, source_path, title, content_hash, created_at, updated_at";

/// Registry entry for a tracked card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardEntry {
    pub slug: String,
    pub note_id: String,
    pub source_path: String,
    pub front: String,
    pub back: String,
    pub content_hash: String,
    pub metadata_hash: String,
    pub language: String,
    pub tags: Vec<String>,
    pub anki_note_id: Option<i64>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
    pub synced_at: Option<DateTime<Utc>>,
}

/// Registry entry for a tracked note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteEntry {
    pub note_id: String,
    pub source_path: String,
    pub title: Option<String>,
    pub content_hash: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Registry errors.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("migration failed from v{from} to v{to}: {reason}")]
    Migration { from: u32, to: u32, reason: String },
    #[error("duplicate slug: {0}")]
    DuplicateSlug(String),
}

/// SQLite-backed card registry with automatic schema migrations.
pub struct CardRegistry {
    conn: rusqlite::Connection,
}

fn tags_to_string(tags: &[String]) -> String {
    tags.join(",")
}

fn string_to_tags(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(String::from).collect()
    }
}

fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.to_utc())
}

fn format_datetime(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

fn row_to_card_entry(row: &rusqlite::Row) -> rusqlite::Result<CardEntry> {
    let tags_str: String = row.get(8)?;
    let created_at_str: Option<String> = row.get(10)?;
    let updated_at_str: Option<String> = row.get(11)?;
    let synced_at_str: Option<String> = row.get(12)?;

    Ok(CardEntry {
        slug: row.get(0)?,
        note_id: row.get(1)?,
        source_path: row.get(2)?,
        front: row.get(3)?,
        back: row.get(4)?,
        content_hash: row.get(5)?,
        metadata_hash: row.get(6)?,
        language: row.get(7)?,
        tags: string_to_tags(&tags_str),
        anki_note_id: row.get(9)?,
        created_at: created_at_str.as_deref().and_then(parse_datetime),
        updated_at: updated_at_str.as_deref().and_then(parse_datetime),
        synced_at: synced_at_str.as_deref().and_then(parse_datetime),
    })
}

fn row_to_note_entry(row: &rusqlite::Row) -> rusqlite::Result<NoteEntry> {
    let created_at_str: Option<String> = row.get(4)?;
    let updated_at_str: Option<String> = row.get(5)?;

    Ok(NoteEntry {
        note_id: row.get(0)?,
        source_path: row.get(1)?,
        title: row.get(2)?,
        content_hash: row.get(3)?,
        created_at: created_at_str.as_deref().and_then(parse_datetime),
        updated_at: updated_at_str.as_deref().and_then(parse_datetime),
    })
}

impl CardRegistry {
    /// Open or create a registry at the given path. Use ":memory:" for tests.
    /// Runs schema migrations automatically.
    pub fn open(db_path: &str) -> Result<Self, RegistryError> {
        let conn = if db_path == ":memory:" {
            rusqlite::Connection::open_in_memory()?
        } else {
            rusqlite::Connection::open(db_path)?
        };

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let registry = Self { conn };
        registry.run_migrations()?;
        Ok(registry)
    }

    /// Close the connection.
    pub fn close(self) {
        drop(self.conn);
    }

    fn run_migrations(&self) -> Result<(), RegistryError> {
        let version = self.get_schema_version()?;

        if version == 0 {
            // Fresh database: create all tables at current schema version
            self.create_schema_v2()?;
        } else if version < SCHEMA_VERSION {
            // Run incremental migrations
            if version < 2 {
                self.migrate_v1_to_v2()?;
            }
        }

        Ok(())
    }

    fn get_schema_version(&self) -> Result<u32, RegistryError> {
        // Check if schema_version table exists
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
            [],
            |row| row.get(0),
        )?;

        if !exists {
            return Ok(0);
        }

        let version: u32 = self.conn.query_row(
            "SELECT version FROM schema_version WHERE id = 1",
            [],
            |row| row.get(0),
        )?;

        Ok(version)
    }

    fn create_schema_v2(&self) -> Result<(), RegistryError> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS cards (
                slug TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                language TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                anki_note_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                synced_at TEXT
            );

            CREATE TABLE IF NOT EXISTS notes (
                note_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            );

            INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, 2);
            ",
        )?;

        Ok(())
    }

    fn migrate_v1_to_v2(&self) -> Result<(), RegistryError> {
        self.conn
            .execute_batch(
                "
            CREATE TABLE IF NOT EXISTS notes (
                note_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            UPDATE schema_version SET version = 2 WHERE id = 1;
            ",
            )
            .map_err(|e| RegistryError::Migration {
                from: 1,
                to: 2,
                reason: e.to_string(),
            })?;

        Ok(())
    }

    // --- Card CRUD ---

    /// Insert a card. Returns Ok(true) on success, Ok(false) if slug exists.
    pub fn add_card(&self, entry: &CardEntry) -> Result<bool, RegistryError> {
        let now = Utc::now();
        let created_at = entry.created_at.unwrap_or(now);
        let updated_at = entry.updated_at.unwrap_or(now);
        let tags_str = tags_to_string(&entry.tags);
        let created_at_str = format_datetime(&created_at);
        let updated_at_str = format_datetime(&updated_at);
        let synced_at_str = entry.synced_at.map(|dt| format_datetime(&dt));

        let result = self.conn.execute(
            "INSERT OR IGNORE INTO cards (slug, note_id, source_path, front, back,
                content_hash, metadata_hash, language, tags, anki_note_id,
                created_at, updated_at, synced_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                entry.slug,
                entry.note_id,
                entry.source_path,
                entry.front,
                entry.back,
                entry.content_hash,
                entry.metadata_hash,
                entry.language,
                tags_str,
                entry.anki_note_id,
                created_at_str,
                updated_at_str,
                synced_at_str,
            ],
        )?;

        Ok(result > 0)
    }

    /// Get card by slug.
    pub fn get_card(&self, slug: &str) -> Result<Option<CardEntry>, RegistryError> {
        let sql = format!("SELECT {CARD_COLUMNS} FROM cards WHERE slug = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![slug], row_to_card_entry)?;
        rows.next().transpose().map_err(Into::into)
    }

    /// Update card by slug. Returns Ok(true) if updated.
    pub fn update_card(&self, entry: &CardEntry) -> Result<bool, RegistryError> {
        let now = Utc::now();
        let updated_at_str = format_datetime(&now);
        let tags_str = tags_to_string(&entry.tags);
        let synced_at_str = entry.synced_at.map(|dt| format_datetime(&dt));

        let result = self.conn.execute(
            "UPDATE cards SET note_id = ?1, source_path = ?2, front = ?3, back = ?4,
                content_hash = ?5, metadata_hash = ?6, language = ?7, tags = ?8,
                anki_note_id = ?9, updated_at = ?10, synced_at = ?11
             WHERE slug = ?12",
            params![
                entry.note_id,
                entry.source_path,
                entry.front,
                entry.back,
                entry.content_hash,
                entry.metadata_hash,
                entry.language,
                tags_str,
                entry.anki_note_id,
                updated_at_str,
                synced_at_str,
                entry.slug,
            ],
        )?;

        Ok(result > 0)
    }

    /// Delete card by slug. Returns Ok(true) if deleted.
    pub fn delete_card(&self, slug: &str) -> Result<bool, RegistryError> {
        let result = self
            .conn
            .execute("DELETE FROM cards WHERE slug = ?1", params![slug])?;
        Ok(result > 0)
    }

    /// Find cards by optional filters (note_id, source_path, content_hash).
    pub fn find_cards(
        &self,
        note_id: Option<&str>,
        source_path: Option<&str>,
        content_hash: Option<&str>,
    ) -> Result<Vec<CardEntry>, RegistryError> {
        let mut sql = format!("SELECT {CARD_COLUMNS} FROM cards WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        for (column, value) in [
            ("note_id", note_id),
            ("source_path", source_path),
            ("content_hash", content_hash),
        ] {
            if let Some(v) = value {
                param_values.push(Box::new(v.to_string()));
                sql.push_str(&format!(" AND {column} = ?{}", param_values.len()));
            }
        }

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), row_to_card_entry)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // --- Note CRUD ---

    /// Insert a note. Returns Ok(true) on success, Ok(false) if note_id exists.
    pub fn add_note(&self, entry: &NoteEntry) -> Result<bool, RegistryError> {
        let now = Utc::now();
        let created_at = entry.created_at.unwrap_or(now);
        let updated_at = entry.updated_at.unwrap_or(now);
        let created_at_str = format_datetime(&created_at);
        let updated_at_str = format_datetime(&updated_at);

        let result = self.conn.execute(
            "INSERT OR IGNORE INTO notes (note_id, source_path, title, content_hash,
                created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entry.note_id,
                entry.source_path,
                entry.title,
                entry.content_hash,
                created_at_str,
                updated_at_str,
            ],
        )?;

        Ok(result > 0)
    }

    /// Get note by note_id.
    pub fn get_note(&self, note_id: &str) -> Result<Option<NoteEntry>, RegistryError> {
        let sql = format!("SELECT {NOTE_COLUMNS} FROM notes WHERE note_id = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![note_id], row_to_note_entry)?;
        rows.next().transpose().map_err(Into::into)
    }

    /// List all notes ordered by note_id.
    pub fn list_notes(&self) -> Result<Vec<NoteEntry>, RegistryError> {
        let sql = format!("SELECT {NOTE_COLUMNS} FROM notes ORDER BY note_id");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], row_to_note_entry)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // --- Stats ---

    /// Return total number of cards.
    pub fn card_count(&self) -> Result<usize, RegistryError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM cards", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Return total number of notes.
    pub fn note_count(&self) -> Result<usize, RegistryError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    // --- Mapping ---

    /// Get all cards for a given note.
    pub fn get_mapping(&self, note_id: &str) -> Result<Vec<CardEntry>, RegistryError> {
        self.find_cards(Some(note_id), None, None)
    }

    /// Replace all cards for a note (delete + re-insert).
    pub fn update_mapping(&self, note_id: &str, cards: &[CardEntry]) -> Result<(), RegistryError> {
        self.conn
            .execute("DELETE FROM cards WHERE note_id = ?1", params![note_id])?;

        for card in cards {
            self.add_card(card)?;
        }

        Ok(())
    }
}
