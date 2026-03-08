use common::error::Result;
use serde_json::Value;
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
    pub fn new(_url: &str, _timeout_secs: u64) -> Self {
        todo!()
    }

    pub async fn invoke(&self, _action: &str, _params: Option<Value>) -> Result<Value> {
        todo!()
    }

    pub async fn ping(&self) -> bool {
        todo!()
    }

    pub async fn version(&self) -> Option<u32> {
        todo!()
    }

    pub async fn deck_names(&self) -> Result<Vec<String>> {
        todo!()
    }

    pub async fn create_deck(&self, _name: &str) -> Result<i64> {
        todo!()
    }

    pub async fn delete_decks(&self, _names: &[String], _cards_too: bool) -> Result<()> {
        todo!()
    }

    pub async fn find_notes(&self, _query: &str) -> Result<Vec<i64>> {
        todo!()
    }

    pub async fn notes_info(&self, _note_ids: &[i64]) -> Result<Vec<Value>> {
        todo!()
    }

    pub async fn add_note(
        &self,
        _deck_name: &str,
        _model_name: &str,
        _fields: &HashMap<String, String>,
        _tags: &[String],
        _allow_duplicate: bool,
    ) -> Result<Option<i64>> {
        todo!()
    }

    pub async fn update_note_fields(
        &self,
        _note_id: i64,
        _fields: &HashMap<String, String>,
    ) -> Result<()> {
        todo!()
    }

    pub async fn delete_notes(&self, _note_ids: &[i64]) -> Result<()> {
        todo!()
    }

    pub async fn get_tags(&self) -> Result<Vec<String>> {
        todo!()
    }

    pub async fn add_tags(&self, _note_ids: &[i64], _tags: &str) -> Result<()> {
        todo!()
    }

    pub async fn remove_tags(&self, _note_ids: &[i64], _tags: &str) -> Result<()> {
        todo!()
    }

    pub async fn model_names(&self) -> Result<Vec<String>> {
        todo!()
    }

    pub async fn model_field_names(&self, _model_name: &str) -> Result<Vec<String>> {
        todo!()
    }

    pub async fn sync(&self) -> Result<()> {
        todo!()
    }
}

impl Default for AnkiConnectClient {
    fn default() -> Self {
        todo!()
    }
}
