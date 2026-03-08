use common::error::{AnkiAtlasError, Result};
use serde_json::{Value, json};
use std::collections::HashMap;

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

impl AnkiConnectClient {
    pub fn new(url: &str, timeout_secs: u64) -> Self {
        Self {
            url: url.to_string(),
            timeout: std::time::Duration::from_secs(timeout_secs),
            client: reqwest::Client::new(),
        }
    }

    /// Send a raw action request to AnkiConnect.
    pub async fn invoke(&self, action: &str, params: Option<Value>) -> Result<Value> {
        let mut payload = json!({
            "action": action,
            "version": ANKI_CONNECT_VERSION,
        });

        if let Some(p) = params {
            payload["params"] = p;
        }

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
                AnkiAtlasError::AnkiConnect {
                    message,
                    context: HashMap::from([("action".into(), action.to_string())]),
                }
            })?;

        let body: Value = response
            .json()
            .await
            .map_err(|e| AnkiAtlasError::AnkiConnect {
                message: format!("failed to parse response: {e}"),
                context: HashMap::from([("action".into(), action.to_string())]),
            })?;

        if let Some(error) = body.get("error").and_then(|e| e.as_str()) {
            return Err(AnkiAtlasError::AnkiConnect {
                message: error.to_string(),
                context: HashMap::from([("action".into(), action.to_string())]),
            });
        }

        Ok(body.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn ping(&self) -> bool {
        self.version().await.is_some()
    }

    pub async fn version(&self) -> Option<u32> {
        self.invoke("version", None)
            .await
            .ok()
            .and_then(|v| v.as_u64().map(|n| n as u32))
    }

    pub async fn deck_names(&self) -> Result<Vec<String>> {
        let result = self.invoke("deckNames", None).await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse deck names: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn create_deck(&self, name: &str) -> Result<i64> {
        let result = self
            .invoke("createDeck", Some(json!({"deck": name})))
            .await?;
        result.as_i64().ok_or_else(|| AnkiAtlasError::AnkiConnect {
            message: "expected deck id".to_string(),
            context: HashMap::new(),
        })
    }

    pub async fn delete_decks(&self, names: &[String], cards_too: bool) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        self.invoke(
            "deleteDecks",
            Some(json!({"decks": names, "cardsToo": cards_too})),
        )
        .await?;
        Ok(())
    }

    pub async fn find_notes(&self, query: &str) -> Result<Vec<i64>> {
        let result = self
            .invoke("findNotes", Some(json!({"query": query})))
            .await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse note ids: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn notes_info(&self, note_ids: &[i64]) -> Result<Vec<Value>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let result = self
            .invoke("notesInfo", Some(json!({"notes": note_ids})))
            .await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse notes info: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn add_note(
        &self,
        deck_name: &str,
        model_name: &str,
        fields: &HashMap<String, String>,
        tags: &[String],
        allow_duplicate: bool,
    ) -> Result<Option<i64>> {
        let mut note = json!({
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "tags": tags,
        });

        if allow_duplicate {
            note["options"] = json!({"allowDuplicate": true});
        }

        match self.invoke("addNote", Some(json!({"note": note}))).await {
            Ok(result) => Ok(result.as_i64()),
            Err(AnkiAtlasError::AnkiConnect { ref message, .. })
                if message.to_lowercase().contains("duplicate") =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    pub async fn update_note_fields(
        &self,
        note_id: i64,
        fields: &HashMap<String, String>,
    ) -> Result<()> {
        self.invoke(
            "updateNoteFields",
            Some(json!({"note": {"id": note_id, "fields": fields}})),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<()> {
        if note_ids.is_empty() {
            return Ok(());
        }
        self.invoke("deleteNotes", Some(json!({"notes": note_ids})))
            .await?;
        Ok(())
    }

    pub async fn get_tags(&self) -> Result<Vec<String>> {
        let result = self.invoke("getTags", None).await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse tags: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn add_tags(&self, note_ids: &[i64], tags: &str) -> Result<()> {
        if note_ids.is_empty() || tags.is_empty() {
            return Ok(());
        }
        self.invoke("addTags", Some(json!({"notes": note_ids, "tags": tags})))
            .await?;
        Ok(())
    }

    pub async fn remove_tags(&self, note_ids: &[i64], tags: &str) -> Result<()> {
        if note_ids.is_empty() || tags.is_empty() {
            return Ok(());
        }
        self.invoke("removeTags", Some(json!({"notes": note_ids, "tags": tags})))
            .await?;
        Ok(())
    }

    pub async fn model_names(&self) -> Result<Vec<String>> {
        let result = self.invoke("modelNames", None).await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse model names: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn model_field_names(&self, model_name: &str) -> Result<Vec<String>> {
        let result = self
            .invoke("modelFieldNames", Some(json!({"modelName": model_name})))
            .await?;
        serde_json::from_value(result).map_err(|e| AnkiAtlasError::AnkiConnect {
            message: format!("failed to parse field names: {e}"),
            context: HashMap::new(),
        })
    }

    pub async fn sync(&self) -> Result<()> {
        self.invoke("sync", None).await?;
        Ok(())
    }
}

impl Default for AnkiConnectClient {
    fn default() -> Self {
        Self::new(ANKI_CONNECT_URL, DEFAULT_TIMEOUT_SECS)
    }
}
