use axum::Json;
use tracing::instrument;

use crate::schemas::{SearchRequest, SearchResponse};

/// Hybrid search across indexed notes. Returns ranked results with scores.
#[instrument(skip(req))]
pub async fn search(Json(req): Json<SearchRequest>) -> Json<SearchResponse> {
    // Stub: real implementation would call search service
    Json(SearchResponse {
        query: req.query,
        results: vec![],
        stats: Default::default(),
        filters_applied: Default::default(),
        lexical: None,
        rerank: None,
    })
}
