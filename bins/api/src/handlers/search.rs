use axum::Json;

use crate::schemas::{SearchRequest, SearchResponse};

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
