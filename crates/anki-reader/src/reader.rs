use crate::models::*;
use chrono::Utc;
use common::error::{AnkiAtlasError, Result};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Read an Anki collection from a SQLite database file.
///
/// Copies the file to a temp location before opening to avoid
/// lock contention with a running Anki instance.
pub struct AnkiReader {
    collection_path: PathBuf,
    conn: Option<Connection>,
    _temp_file: Option<NamedTempFile>,
}

impl AnkiReader {
    /// Create a new reader. Validates that the collection file exists.
    pub fn new(collection_path: impl AsRef<Path>) -> Result<Self> {
        let path = collection_path.as_ref();
        if !path.exists() {
            return Err(AnkiAtlasError::AnkiReader {
                message: format!("collection file not found: {}", path.display()),
                context: HashMap::new(),
            });
        }
        Ok(Self {
            collection_path: path.to_path_buf(),
            conn: None,
            _temp_file: None,
        })
    }

    /// Open the database (copy to temp, connect).
    pub fn open(&mut self) -> Result<()> {
        let temp_file = NamedTempFile::new().map_err(|e| AnkiAtlasError::AnkiReader {
            message: format!("failed to create temp file: {e}"),
            context: HashMap::new(),
        })?;

        std::fs::copy(&self.collection_path, temp_file.path()).map_err(|e| {
            AnkiAtlasError::AnkiReader {
                message: format!("failed to copy collection: {e}"),
                context: HashMap::new(),
            }
        })?;

        let conn =
            Connection::open(temp_file.path()).map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to open database: {e}"),
                context: HashMap::new(),
            })?;

        self.conn = Some(conn);
        self._temp_file = Some(temp_file);
        Ok(())
    }

    /// Close the database and clean up temp file.
    pub fn close(&mut self) {
        self.conn = None;
        self._temp_file = None;
    }

    fn conn(&self) -> Result<&Connection> {
        self.conn.as_ref().ok_or_else(|| AnkiAtlasError::AnkiReader {
            message: "reader not opened".to_string(),
            context: HashMap::new(),
        })
    }

    fn has_modern_schema(&self) -> Result<bool> {
        let conn = self.conn()?;
        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='notetypes'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("schema detection failed: {e}"),
                context: HashMap::new(),
            })?;
        Ok(count > 0)
    }

    /// Read the complete collection.
    pub fn read_collection(&self) -> Result<AnkiCollection> {
        let decks = self.read_decks()?;
        let models = self.read_models()?;
        let notes = self.read_notes(&models)?;
        let cards = self.read_cards()?;
        let card_stats = self.compute_card_stats()?;

        Ok(AnkiCollection {
            decks,
            models,
            notes,
            cards,
            card_stats,
            collection_path: Some(self.collection_path.to_string_lossy().to_string()),
            extracted_at: Utc::now(),
            schema_version: 11,
        })
    }

    /// Read all decks.
    pub fn read_decks(&self) -> Result<Vec<AnkiDeck>> {
        if self.has_modern_schema()? {
            self.read_decks_modern()
        } else {
            self.read_decks_legacy()
        }
    }

    fn read_decks_legacy(&self) -> Result<Vec<AnkiDeck>> {
        let conn = self.conn()?;
        let decks_json: String =
            conn.query_row("SELECT decks FROM col", [], |row| row.get(0))
                .map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to read decks: {e}"),
                    context: HashMap::new(),
                })?;

        let decks_map: HashMap<String, serde_json::Value> =
            serde_json::from_str(&decks_json).map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to parse decks JSON: {e}"),
                context: HashMap::new(),
            })?;

        let mut decks = Vec::new();
        for (_key, val) in decks_map {
            let deck_id = val["id"].as_i64().unwrap_or(0);
            let name = val["name"].as_str().unwrap_or("").to_string();
            let parent_name = if name.contains("::") {
                name.rsplit_once("::").map(|(parent, _)| parent.to_string())
            } else {
                None
            };
            decks.push(AnkiDeck {
                deck_id,
                name,
                parent_name,
                config: val.clone(),
            });
        }
        Ok(decks)
    }

    fn read_decks_modern(&self) -> Result<Vec<AnkiDeck>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, name FROM decks")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query decks: {e}"),
                context: HashMap::new(),
            })?;

        let decks = stmt
            .query_map([], |row| {
                let deck_id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let parent_name = if name.contains("::") {
                    name.rsplit_once("::").map(|(parent, _)| parent.to_string())
                } else {
                    None
                };
                Ok(AnkiDeck {
                    deck_id,
                    name,
                    parent_name,
                    config: serde_json::Value::Null,
                })
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read decks: {e}"),
                context: HashMap::new(),
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to collect decks: {e}"),
                context: HashMap::new(),
            })?;

        Ok(decks)
    }

    /// Read all note types/models.
    pub fn read_models(&self) -> Result<Vec<AnkiModel>> {
        if self.has_modern_schema()? {
            self.read_models_modern()
        } else {
            self.read_models_legacy()
        }
    }

    fn read_models_legacy(&self) -> Result<Vec<AnkiModel>> {
        let conn = self.conn()?;
        let models_json: String =
            conn.query_row("SELECT models FROM col", [], |row| row.get(0))
                .map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to read models: {e}"),
                    context: HashMap::new(),
                })?;

        let models_map: HashMap<String, serde_json::Value> =
            serde_json::from_str(&models_json).map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to parse models JSON: {e}"),
                context: HashMap::new(),
            })?;

        let mut models = Vec::new();
        for (_key, val) in models_map {
            let model_id = val["id"].as_i64().unwrap_or(0);
            let name = val["name"].as_str().unwrap_or("").to_string();
            let fields = val["flds"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let templates = val["tmpls"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            models.push(AnkiModel {
                model_id,
                name,
                fields,
                templates,
                config: val.clone(),
            });
        }
        Ok(models)
    }

    fn read_models_modern(&self) -> Result<Vec<AnkiModel>> {
        let conn = self.conn()?;

        // Read fields grouped by notetype
        let mut field_stmt = conn
            .prepare("SELECT ntid, name, ord FROM fields ORDER BY ntid, ord")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query fields: {e}"),
                context: HashMap::new(),
            })?;

        let mut model_fields: HashMap<i64, Vec<serde_json::Value>> = HashMap::new();
        let rows = field_stmt
            .query_map([], |row| {
                let ntid: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let ord: i32 = row.get(2)?;
                Ok((ntid, name, ord))
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read fields: {e}"),
                context: HashMap::new(),
            })?;

        for row in rows {
            let (ntid, name, ord) = row.map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read field row: {e}"),
                context: HashMap::new(),
            })?;
            model_fields
                .entry(ntid)
                .or_default()
                .push(serde_json::json!({"name": name, "ord": ord}));
        }

        // Read templates if table exists
        let mut model_templates: HashMap<i64, Vec<serde_json::Value>> = HashMap::new();
        let has_templates: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='templates'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if has_templates > 0 {
            let mut tmpl_stmt = conn
                .prepare("SELECT ntid, name, ord FROM templates ORDER BY ntid, ord")
                .map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to query templates: {e}"),
                    context: HashMap::new(),
                })?;

            let rows = tmpl_stmt
                .query_map([], |row| {
                    let ntid: i64 = row.get(0)?;
                    let name: String = row.get(1)?;
                    let ord: i32 = row.get(2)?;
                    Ok((ntid, name, ord))
                })
                .map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to read templates: {e}"),
                    context: HashMap::new(),
                })?;

            for row in rows {
                let (ntid, name, ord) = row.map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to read template row: {e}"),
                    context: HashMap::new(),
                })?;
                model_templates
                    .entry(ntid)
                    .or_default()
                    .push(serde_json::json!({"name": name, "ord": ord}));
            }
        }

        // Read notetypes
        let mut stmt = conn
            .prepare("SELECT id, name FROM notetypes")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query notetypes: {e}"),
                context: HashMap::new(),
            })?;

        let models = stmt
            .query_map([], |row| {
                let model_id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                Ok((model_id, name))
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read notetypes: {e}"),
                context: HashMap::new(),
            })?
            .map(|r| {
                let (model_id, name) = r.map_err(|e| AnkiAtlasError::AnkiReader {
                    message: format!("failed to read notetype row: {e}"),
                    context: HashMap::new(),
                })?;
                Ok(AnkiModel {
                    model_id,
                    name,
                    fields: model_fields.remove(&model_id).unwrap_or_default(),
                    templates: model_templates.remove(&model_id).unwrap_or_default(),
                    config: serde_json::Value::Null,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(models)
    }

    /// Read all notes.
    pub fn read_notes(&self, models: &[AnkiModel]) -> Result<Vec<AnkiNote>> {
        let conn = self.conn()?;

        // Build model_id -> field names mapping
        let mut model_field_names: HashMap<i64, Vec<String>> = HashMap::new();
        for model in models {
            let names: Vec<String> = model
                .fields
                .iter()
                .filter_map(|f| f.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect();
            model_field_names.insert(model.model_id, names);
        }

        let mut stmt = conn
            .prepare("SELECT id, mid, tags, flds, mod, usn FROM notes")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query notes: {e}"),
                context: HashMap::new(),
            })?;

        let notes = stmt
            .query_map([], |row| {
                let note_id: i64 = row.get(0)?;
                let model_id: i64 = row.get(1)?;
                let tags_str: String = row.get(2)?;
                let fields_str: String = row.get(3)?;
                let mtime: i64 = row.get(4)?;
                let usn: i32 = row.get(5)?;
                Ok((note_id, model_id, tags_str, fields_str, mtime, usn))
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read notes: {e}"),
                context: HashMap::new(),
            })?
            .map(|r| {
                let (note_id, model_id, tags_str, fields_str, mtime, usn) =
                    r.map_err(|e| AnkiAtlasError::AnkiReader {
                        message: format!("failed to read note row: {e}"),
                        context: HashMap::new(),
                    })?;

                let tags: Vec<String> = tags_str
                    .split_whitespace()
                    .map(String::from)
                    .collect();

                let fields: Vec<String> =
                    fields_str.split('\x1f').map(String::from).collect();

                let field_names = model_field_names.get(&model_id);
                let fields_json: HashMap<String, String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, val)| {
                        let name = field_names
                            .and_then(|names| names.get(i))
                            .cloned()
                            .unwrap_or_else(|| format!("Field{i}"));
                        (name, val.clone())
                    })
                    .collect();

                Ok(AnkiNote {
                    note_id,
                    model_id,
                    tags,
                    fields,
                    fields_json,
                    raw_fields: Some(fields_str),
                    normalized_text: String::new(),
                    mtime,
                    usn,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(notes)
    }

    /// Read all cards.
    pub fn read_cards(&self) -> Result<Vec<AnkiCard>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses FROM cards")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query cards: {e}"),
                context: HashMap::new(),
            })?;

        let cards = stmt
            .query_map([], |row| {
                Ok(AnkiCard {
                    card_id: row.get(0)?,
                    note_id: row.get(1)?,
                    deck_id: row.get(2)?,
                    ord: row.get(3)?,
                    mtime: row.get(4)?,
                    usn: row.get(5)?,
                    card_type: row.get(6)?,
                    queue: row.get(7)?,
                    due: row.get(8)?,
                    ivl: row.get(9)?,
                    ease: row.get(10)?,
                    reps: row.get(11)?,
                    lapses: row.get(12)?,
                })
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read cards: {e}"),
                context: HashMap::new(),
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to collect cards: {e}"),
                context: HashMap::new(),
            })?;

        Ok(cards)
    }

    /// Read review log entries.
    pub fn read_revlog(&self) -> Result<Vec<AnkiRevlogEntry>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, cid, usn, ease, ivl, lastIvl, factor, time, type FROM revlog")
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query revlog: {e}"),
                context: HashMap::new(),
            })?;

        let entries = stmt
            .query_map([], |row| {
                Ok(AnkiRevlogEntry {
                    id: row.get(0)?,
                    card_id: row.get(1)?,
                    usn: row.get(2)?,
                    button_chosen: row.get(3)?,
                    interval: row.get(4)?,
                    last_interval: row.get(5)?,
                    ease: row.get(6)?,
                    time_ms: row.get(7)?,
                    review_type: row.get(8)?,
                })
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read revlog: {e}"),
                context: HashMap::new(),
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to collect revlog: {e}"),
                context: HashMap::new(),
            })?;

        Ok(entries)
    }

    /// Compute aggregated card statistics from the revlog.
    pub fn compute_card_stats(&self) -> Result<Vec<CardStats>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT
                    cid as card_id,
                    COUNT(*) as reviews,
                    AVG(factor) as avg_ease,
                    SUM(CASE WHEN ease = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as fail_rate,
                    MAX(id) as last_review_ms,
                    SUM(time) as total_time_ms
                FROM revlog
                GROUP BY cid",
            )
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to query card stats: {e}"),
                context: HashMap::new(),
            })?;

        let stats = stmt
            .query_map([], |row| {
                let card_id: i64 = row.get(0)?;
                let reviews: i32 = row.get(1)?;
                let avg_ease: Option<f64> = row.get(2)?;
                let fail_rate: Option<f64> = row.get(3)?;
                let last_review_ms: Option<i64> = row.get(4)?;
                let total_time_ms: i64 = row.get::<_, Option<i64>>(5)?.unwrap_or(0);

                let last_review_at = last_review_ms.map(|ms| {
                    chrono::DateTime::from_timestamp(ms / 1000, 0)
                        .unwrap_or_default()
                        .with_timezone(&Utc)
                });

                Ok(CardStats {
                    card_id,
                    reviews,
                    avg_ease,
                    fail_rate,
                    last_review_at,
                    total_time_ms,
                })
            })
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to read card stats: {e}"),
                context: HashMap::new(),
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| AnkiAtlasError::AnkiReader {
                message: format!("failed to collect card stats: {e}"),
                context: HashMap::new(),
            })?;

        Ok(stats)
    }
}

impl Drop for AnkiReader {
    fn drop(&mut self) {
        self.close();
    }
}

/// Convenience function: open, read, close.
pub fn read_anki_collection(path: impl AsRef<Path>) -> Result<AnkiCollection> {
    let mut reader = AnkiReader::new(path)?;
    reader.open()?;
    let collection = reader.read_collection()?;
    reader.close();
    Ok(collection)
}
