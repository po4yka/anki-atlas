use serde::{Deserialize, Serialize};

pub const PROMPT_VERSION: &str = "apf-v2.1";

/// Input spec for rendering a card to APF HTML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardSpec {
    pub card_index: u32,
    pub slug: String,
    pub slug_base: Option<String>,
    pub lang: String,
    pub card_type: String,
    pub tags: Vec<String>,
    pub guid: String,
    pub source_path: Option<String>,
    pub source_anchor: Option<String>,
    pub title: String,
    pub key_point_code: Option<String>,
    pub key_point_code_lang: Option<String>,
    pub key_point_notes: Vec<String>,
    pub other_notes: Option<String>,
    pub extra: Option<String>,
}
