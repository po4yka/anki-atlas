use common::error::{AnkiAtlasError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use tracing::instrument;

/// Default AnkiConnect URL.
pub const ANKI_CONNECT_URL: &str = "http://localhost:8765";
/// AnkiConnect protocol version.
pub const ANKI_CONNECT_VERSION: u32 = 6;
/// Default request timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Async HTTP client for the AnkiConnect API.
pub struct AnkiConnectClient {
    url: String,
    timeout: std::time::Duration,
    client: reqwest::Client,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AnkiConnectNoteField {
    pub value: String,
    pub order: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AnkiConnectNoteInfo {
    #[serde(rename = "noteId")]
    pub note_id: i64,
    #[serde(rename = "modelName")]
    pub model_name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub fields: HashMap<String, AnkiConnectNoteField>,
    #[serde(default)]
    pub cards: Vec<i64>,
}

/// Controls duplicate note handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateHandling {
    Allow,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddNoteOutcome {
    Added(i64),
    Duplicate,
}

#[derive(Serialize)]
struct AnkiConnectRequest<'a> {
    action: &'a str,
    version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

impl AnkiConnectClient {
    pub fn new(url: &str, timeout_secs: u64) -> Self {
        Self {
            url: url.to_string(),
            timeout: std::time::Duration::from_secs(timeout_secs),
            client: reqwest::Client::new(),
        }
    }

    fn action_error(&self, action: &str, message: impl Into<String>) -> AnkiAtlasError {
        AnkiAtlasError::AnkiConnect {
            message: message.into(),
            context: HashMap::from([("action".into(), action.to_string())]),
        }
    }

    async fn invoke_value(&self, action: &str, params: Option<Value>) -> Result<Value> {
        let payload = AnkiConnectRequest {
            action,
            version: ANKI_CONNECT_VERSION,
            params,
        };

        let response = self
            .client
            .post(&self.url)
            .json(&payload)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                let message = if e.is_connect() {
                    format!(
                        "AnkiConnect not reachable at {}. Is Anki running?",
                        self.url
                    )
                } else {
                    format!("HTTP error: {e}")
                };
                self.action_error(action, message)
            })?;

        let body: Value = response
            .json()
            .await
            .map_err(|e| self.action_error(action, format!("failed to parse response: {e}")))?;

        if let Some(error) = body.get("error").and_then(|e| e.as_str()) {
            return Err(self.action_error(action, error));
        }

        body.get("result")
            .cloned()
            .ok_or_else(|| self.action_error(action, "response missing result field"))
    }

    async fn invoke_typed<T: DeserializeOwned>(
        &self,
        action: &str,
        params: Option<Value>,
    ) -> Result<T> {
        let result = self.invoke_value(action, params).await?;
        serde_json::from_value(result)
            .map_err(|e| self.action_error(action, format!("failed to parse result: {e}")))
    }

    async fn invoke_unit(&self, action: &str, params: Option<Value>) -> Result<()> {
        let result = self.invoke_value(action, params).await?;
        if result.is_null() {
            Ok(())
        } else {
            Err(self.action_error(action, "expected null result"))
        }
    }

    #[instrument(skip(self))]
    pub async fn ping(&self) -> Result<()> {
        self.version().await.map(|_| ())
    }

    #[instrument(skip(self))]
    pub async fn version(&self) -> Result<u32> {
        self.invoke_typed("version", None).await
    }

    #[instrument(skip(self))]
    pub async fn deck_names(&self) -> Result<Vec<String>> {
        self.invoke_typed("deckNames", None).await
    }

    #[instrument(skip(self))]
    pub async fn create_deck(&self, name: &str) -> Result<i64> {
        self.invoke_typed("createDeck", Some(json!({"deck": name})))
            .await
    }

    #[instrument(skip(self))]
    pub async fn delete_decks(&self, names: &[String], cards_too: bool) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        self.invoke_unit(
            "deleteDecks",
            Some(json!({"decks": names, "cardsToo": cards_too})),
        )
        .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn find_notes(&self, query: &str) -> Result<Vec<i64>> {
        self.invoke_typed("findNotes", Some(json!({"query": query})))
            .await
    }

    #[instrument(skip(self))]
    pub async fn notes_info(&self, note_ids: &[i64]) -> Result<Vec<AnkiConnectNoteInfo>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.invoke_typed("notesInfo", Some(json!({"notes": note_ids})))
            .await
    }

    #[instrument(skip(self, fields, tags))]
    pub async fn add_note(
        &self,
        deck_name: &str,
        model_name: &str,
        fields: &HashMap<String, String>,
        tags: &[String],
        allow_duplicate: DuplicateHandling,
    ) -> Result<AddNoteOutcome> {
        let mut note = json!({
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "tags": tags,
        });

        if allow_duplicate == DuplicateHandling::Allow {
            note["options"] = json!({"allowDuplicate": true});
        }

        match self
            .invoke_typed("addNote", Some(json!({"note": note})))
            .await
        {
            Ok(note_id) => Ok(AddNoteOutcome::Added(note_id)),
            Err(AnkiAtlasError::AnkiConnect { ref message, .. })
                if message.to_lowercase().contains("duplicate") =>
            {
                Ok(AddNoteOutcome::Duplicate)
            }
            Err(e) => Err(e),
        }
    }

    #[instrument(skip(self, fields))]
    pub async fn update_note_fields(
        &self,
        note_id: i64,
        fields: &HashMap<String, String>,
    ) -> Result<()> {
        self.invoke_unit(
            "updateNoteFields",
            Some(json!({"note": {"id": note_id, "fields": fields}})),
        )
        .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<()> {
        if note_ids.is_empty() {
            return Ok(());
        }
        self.invoke_unit("deleteNotes", Some(json!({"notes": note_ids})))
            .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn get_tags(&self) -> Result<Vec<String>> {
        self.invoke_typed("getTags", None).await
    }

    #[instrument(skip(self, note_ids))]
    pub async fn add_tags(&self, note_ids: &[i64], tags: &str) -> Result<()> {
        if note_ids.is_empty() || tags.is_empty() {
            return Ok(());
        }
        self.invoke_unit("addTags", Some(json!({"notes": note_ids, "tags": tags})))
            .await?;
        Ok(())
    }

    #[instrument(skip(self, note_ids))]
    pub async fn remove_tags(&self, note_ids: &[i64], tags: &str) -> Result<()> {
        if note_ids.is_empty() || tags.is_empty() {
            return Ok(());
        }
        self.invoke_unit("removeTags", Some(json!({"notes": note_ids, "tags": tags})))
            .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn model_names(&self) -> Result<Vec<String>> {
        self.invoke_typed("modelNames", None).await
    }

    #[instrument(skip(self))]
    pub async fn model_field_names(&self, model_name: &str) -> Result<Vec<String>> {
        self.invoke_typed("modelFieldNames", Some(json!({"modelName": model_name})))
            .await
    }

    #[instrument(skip(self))]
    pub async fn sync(&self) -> Result<()> {
        self.invoke_unit("sync", None).await?;
        Ok(())
    }
}

impl Default for AnkiConnectClient {
    fn default() -> Self {
        Self::new(ANKI_CONNECT_URL, DEFAULT_TIMEOUT_SECS)
    }
}
