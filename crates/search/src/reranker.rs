use crate::error::SearchError;

/// Trait for second-stage reranking implementations.
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait Reranker: Send + Sync {
    /// Score (document_id, text) pairs against a query.
    /// Returns (document_id, score) pairs in arbitrary order.
    async fn rerank(
        &self,
        query: &str,
        documents: &[(i64, String)],
    ) -> Result<Vec<(i64, f64)>, SearchError>;
}

/// Cross-encoder reranker that calls an external inference endpoint.
#[allow(dead_code)]
pub struct CrossEncoderReranker {
    model_name: String,
    batch_size: usize,
    client: reqwest::Client,
    endpoint: String,
}

impl CrossEncoderReranker {
    pub fn new(
        model_name: impl Into<String>,
        batch_size: usize,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            batch_size,
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
        }
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[async_trait::async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[(i64, String)],
    ) -> Result<Vec<(i64, f64)>, SearchError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────

    #[test]
    fn new_creates_reranker_with_correct_fields() {
        let reranker =
            CrossEncoderReranker::new("cross-encoder/ms-marco-MiniLM-L-6-v2", 32, "http://localhost:8080/rerank");
        assert_eq!(reranker.model_name(), "cross-encoder/ms-marco-MiniLM-L-6-v2");
    }

    #[test]
    fn new_accepts_string_types() {
        let model = String::from("my-model");
        let endpoint = String::from("http://example.com/rerank");
        let reranker = CrossEncoderReranker::new(model, 16, endpoint);
        assert_eq!(reranker.model_name(), "my-model");
    }

    // ── Reranker trait on CrossEncoderReranker ─────────────────────

    #[tokio::test]
    async fn rerank_empty_documents_returns_empty() {
        let reranker =
            CrossEncoderReranker::new("test-model", 32, "http://localhost:8080/rerank");
        let result = reranker.rerank("test query", &[]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ── Send + Sync ──────────────────────────────────────────────

    #[test]
    fn cross_encoder_reranker_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CrossEncoderReranker>();
    }

    #[test]
    fn reranker_trait_object_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn Reranker>>();
    }

    // ── MockReranker ──────────────────────────────────────────────

    #[tokio::test]
    async fn mock_reranker_returns_configured_scores() {
        let mut mock = MockReranker::new();
        mock.expect_rerank()
            .returning(|_query, docs| {
                let scores: Vec<(i64, f64)> = docs
                    .iter()
                    .enumerate()
                    .map(|(i, (id, _))| (*id, 1.0 - i as f64 * 0.1))
                    .collect();
                Box::pin(async move { Ok(scores) })
            });

        let docs = vec![
            (1, "first document".to_string()),
            (2, "second document".to_string()),
            (3, "third document".to_string()),
        ];
        let result = mock.rerank("test query", &docs).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (1, 1.0));
        assert_eq!(result[1], (2, 0.9));
        assert_eq!(result[2], (3, 0.8));
    }

    #[tokio::test]
    async fn mock_reranker_can_return_error() {
        let mut mock = MockReranker::new();
        mock.expect_rerank()
            .returning(|_query, _docs| {
                Box::pin(async {
                    Err(SearchError::Rerank("model unavailable".to_string()))
                })
            });

        let docs = vec![(1, "doc".to_string())];
        let result = mock.rerank("query", &docs).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SearchError::Rerank(_)));
    }

    #[tokio::test]
    async fn mock_reranker_preserves_document_ids() {
        let mut mock = MockReranker::new();
        mock.expect_rerank()
            .returning(|_query, docs| {
                let scores: Vec<(i64, f64)> =
                    docs.iter().map(|(id, _)| (*id, 0.5)).collect();
                Box::pin(async move { Ok(scores) })
            });

        let docs = vec![
            (42, "answer to everything".to_string()),
            (99, "last doc".to_string()),
        ];
        let result = mock.rerank("query", &docs).await.unwrap();
        let ids: Vec<i64> = result.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&42));
        assert!(ids.contains(&99));
    }

    #[tokio::test]
    async fn rerank_returns_scores_in_arbitrary_order() {
        let mut mock = MockReranker::new();
        mock.expect_rerank()
            .returning(|_query, docs| {
                let scores: Vec<(i64, f64)> =
                    docs.iter().rev().map(|(id, _)| (*id, 0.5)).collect();
                Box::pin(async move { Ok(scores) })
            });

        let docs = vec![
            (1, "first".to_string()),
            (2, "second".to_string()),
        ];
        let result = mock.rerank("query", &docs).await.unwrap();
        assert_eq!(result.len(), 2);
        let ids: Vec<i64> = result.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }
}
