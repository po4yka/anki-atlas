use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::RagError;
use crate::store::{MetadataFilter, SearchResult, VectorStore};

/// A related concept from the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedConcept {
    pub title: String,
    pub content: String,
    pub topic: String,
    pub similarity: f32,
    pub source_file: String,
}

/// Result of a duplicate detection check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateCheckResult {
    pub is_duplicate: bool,
    pub confidence: f32,
    pub similar_items: Vec<SearchResult>,
    pub recommendation: String,
}

/// A few-shot example for card generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub question: String,
    pub answer: String,
    pub topic: String,
    pub difficulty: String,
    pub source_file: String,
}

/// High-level RAG operations for flashcard generation.
pub struct RagService<S: VectorStore> {
    #[allow(dead_code)] // used in future GREEN phase
    store: Arc<S>,
}

impl<S: VectorStore> RagService<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    /// Check whether a card is a potential duplicate.
    pub async fn find_duplicates(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Result<DuplicateCheckResult, RagError> {
        let results = self.store.search(query_embedding, k, None, 0.0).await?;

        if results.is_empty() {
            return Ok(DuplicateCheckResult {
                is_duplicate: false,
                confidence: 0.0,
                similar_items: Vec::new(),
                recommendation: "No significant duplicates found".to_string(),
            });
        }

        let max_similarity = results.iter().map(|r| r.similarity()).fold(0.0_f32, f32::max);

        let recommendation = if max_similarity >= 0.95 {
            "Highly likely duplicate -- skip this card"
        } else if max_similarity >= 0.85 {
            "Probable duplicate -- review before creating"
        } else if max_similarity >= 0.70 {
            "Similar content exists -- consider differentiating"
        } else {
            "No significant duplicates found"
        };

        Ok(DuplicateCheckResult {
            is_duplicate: max_similarity >= threshold,
            confidence: max_similarity,
            similar_items: results,
            recommendation: recommendation.to_string(),
        })
    }

    /// Retrieve related concepts for context enrichment.
    pub async fn get_context(
        &self,
        query_embedding: &[f32],
        k: usize,
        topic: Option<&str>,
        min_similarity: f32,
    ) -> Result<Vec<RelatedConcept>, RagError> {
        let filter = topic.map(|t| MetadataFilter {
            field: "topic".to_string(),
            value: t.to_string(),
        });

        let results = self
            .store
            .search(query_embedding, k * 2, filter, min_similarity)
            .await?;

        let mut seen = HashSet::new();
        let concepts = results
            .into_iter()
            .filter(|r| seen.insert(r.source_file.clone()))
            .take(k)
            .map(|r| {
                let similarity = r.similarity();
                RelatedConcept {
                    title: r.metadata.get("title").cloned().unwrap_or_default(),
                    content: r.content,
                    topic: r.metadata.get("topic").cloned().unwrap_or_default(),
                    similarity,
                    source_file: r.source_file,
                }
            })
            .collect();

        Ok(concepts)
    }

    /// Retrieve few-shot examples for generation prompts.
    pub async fn get_few_shot_examples(
        &self,
        query_embedding: &[f32],
        k: usize,
        topic: Option<&str>,
    ) -> Result<Vec<FewShotExample>, RagError> {
        let filter = topic.map(|t| MetadataFilter {
            field: "topic".to_string(),
            value: t.to_string(),
        });

        let results = self
            .store
            .search(query_embedding, k * 3, filter, 0.0)
            .await?;

        let examples = results
            .into_iter()
            .take(k)
            .map(|r| {
                let question = truncate(&r.content, 300);
                let answer = truncate(&r.content, 500);
                FewShotExample {
                    question,
                    answer,
                    topic: r.metadata.get("topic").cloned().unwrap_or_default(),
                    difficulty: r.metadata.get("difficulty").cloned().unwrap_or_default(),
                    source_file: r.source_file,
                }
            })
            .collect();

        Ok(examples)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        s[..max].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MockVectorStore;
    use std::collections::HashMap;

    fn make_search_result(id: &str, content: &str, score: f32, source: &str) -> SearchResult {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), format!("Title for {id}"));
        metadata.insert("topic".to_string(), "rust".to_string());
        metadata.insert("difficulty".to_string(), "medium".to_string());
        SearchResult {
            chunk_id: id.to_string(),
            content: content.to_string(),
            score,
            source_file: source.to_string(),
            metadata,
        }
    }

    fn make_service(mock: MockVectorStore) -> RagService<MockVectorStore> {
        RagService::new(Arc::new(mock))
    }

    // ===== find_duplicates =====

    #[tokio::test]
    async fn find_duplicates_returns_duplicate_when_results_above_threshold() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            Ok(vec![make_search_result("c1", "content", 0.02, "f.md")])
            // score=0.02 -> similarity = 1/(1+0.02) ~= 0.98
        });

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0, 2.0], 0.85, 5).await.unwrap();

        assert!(result.is_duplicate, "Should detect duplicate when results exist");
        assert!(result.confidence > 0.85);
        assert_eq!(result.similar_items.len(), 1);
    }

    #[tokio::test]
    async fn find_duplicates_returns_no_duplicate_when_no_results() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0, 2.0], 0.85, 5).await.unwrap();

        assert!(!result.is_duplicate);
        assert!((result.confidence - 0.0).abs() < f32::EPSILON);
        assert!(result.similar_items.is_empty());
    }

    #[tokio::test]
    async fn find_duplicates_recommendation_highly_likely() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            // score ~0.01 -> similarity ~0.99 -> >= 0.95
            Ok(vec![make_search_result("c1", "dup", 0.01, "f.md")])
        });

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0], 0.85, 5).await.unwrap();

        assert_eq!(result.recommendation, "Highly likely duplicate -- skip this card");
    }

    #[tokio::test]
    async fn find_duplicates_recommendation_probable() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            // score ~0.10 -> similarity = 1/1.10 ~= 0.909 -> >= 0.85
            Ok(vec![make_search_result("c1", "dup", 0.10, "f.md")])
        });

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0], 0.85, 5).await.unwrap();

        assert_eq!(
            result.recommendation,
            "Probable duplicate -- review before creating"
        );
    }

    #[tokio::test]
    async fn find_duplicates_recommendation_similar() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            // score ~0.30 -> similarity = 1/1.30 ~= 0.769 -> >= 0.70
            Ok(vec![make_search_result("c1", "sim", 0.30, "f.md")])
        });

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0], 0.85, 5).await.unwrap();

        assert_eq!(
            result.recommendation,
            "Similar content exists -- consider differentiating"
        );
    }

    #[tokio::test]
    async fn find_duplicates_recommendation_no_duplicates() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            // score ~1.0 -> similarity = 0.5 -> < 0.70
            Ok(vec![make_search_result("c1", "diff", 1.0, "f.md")])
        });

        let svc = make_service(mock);
        let result = svc.find_duplicates(&[1.0], 0.85, 5).await.unwrap();

        assert_eq!(result.recommendation, "No significant duplicates found");
    }

    // ===== get_context =====

    #[tokio::test]
    async fn get_context_deduplicates_by_source_file() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            Ok(vec![
                make_search_result("c1", "first", 0.1, "a.md"),
                make_search_result("c2", "second from a", 0.2, "a.md"), // same source
                make_search_result("c3", "from b", 0.3, "b.md"),
            ])
        });

        let svc = make_service(mock);
        let concepts = svc.get_context(&[1.0], 5, None, 0.3).await.unwrap();

        // Should keep only first occurrence per source_file
        let sources: Vec<&str> = concepts.iter().map(|c| c.source_file.as_str()).collect();
        assert_eq!(sources.len(), 2, "Should dedup by source_file");
        assert!(sources.contains(&"a.md"));
        assert!(sources.contains(&"b.md"));
    }

    #[tokio::test]
    async fn get_context_returns_at_most_k_results() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().times(1).returning(|_, _, _, _| {
            Ok(vec![
                make_search_result("c1", "one", 0.1, "a.md"),
                make_search_result("c2", "two", 0.2, "b.md"),
                make_search_result("c3", "three", 0.3, "c.md"),
                make_search_result("c4", "four", 0.4, "d.md"),
                make_search_result("c5", "five", 0.5, "e.md"),
            ])
        });

        let svc = make_service(mock);
        let concepts = svc.get_context(&[1.0], 2, None, 0.3).await.unwrap();

        assert!(concepts.len() <= 2, "Should return at most k=2 results, got {}", concepts.len());
    }

    #[tokio::test]
    async fn get_context_fetches_k_times_2_from_store() {
        let mut mock = MockVectorStore::new();
        mock.expect_search()
            .times(1)
            .withf(|_, k, _, _| *k == 10) // k=5 -> fetches 5*2=10
            .returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let _ = svc.get_context(&[1.0], 5, None, 0.3).await.unwrap();
    }

    #[tokio::test]
    async fn get_context_passes_topic_filter() {
        let mut mock = MockVectorStore::new();
        mock.expect_search()
            .times(1)
            .withf(|_, _, filter, _| {
                filter
                    .as_ref()
                    .map_or(false, |f| f.field == "topic" && f.value == "rust")
            })
            .returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let _ = svc
            .get_context(&[1.0], 5, Some("rust"), 0.3)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn get_context_no_filter_when_topic_none() {
        let mut mock = MockVectorStore::new();
        mock.expect_search()
            .times(1)
            .withf(|_, _, filter: &Option<MetadataFilter>, _| filter.is_none())
            .returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let _ = svc.get_context(&[1.0], 5, None, 0.3).await.unwrap();
    }

    #[tokio::test]
    async fn get_context_maps_metadata_to_concept_fields() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            Ok(vec![make_search_result("c1", "content", 0.1, "a.md")])
        });

        let svc = make_service(mock);
        let concepts = svc.get_context(&[1.0], 5, None, 0.3).await.unwrap();

        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].title, "Title for c1");
        assert_eq!(concepts[0].topic, "rust");
        assert_eq!(concepts[0].content, "content");
        assert_eq!(concepts[0].source_file, "a.md");
    }

    // ===== get_few_shot_examples =====

    #[tokio::test]
    async fn get_few_shot_examples_truncates_question_to_300() {
        let long_content = "x".repeat(600);
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(move |_, _, _, _| {
            Ok(vec![make_search_result("c1", &long_content, 0.1, "f.md")])
        });

        let svc = make_service(mock);
        let examples = svc.get_few_shot_examples(&[1.0], 3, None).await.unwrap();

        assert_eq!(examples.len(), 1);
        assert!(
            examples[0].question.len() <= 300,
            "Question should be truncated to 300 chars, got {}",
            examples[0].question.len()
        );
    }

    #[tokio::test]
    async fn get_few_shot_examples_truncates_answer_to_500() {
        let long_content = "y".repeat(600);
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(move |_, _, _, _| {
            Ok(vec![make_search_result("c1", &long_content, 0.1, "f.md")])
        });

        let svc = make_service(mock);
        let examples = svc.get_few_shot_examples(&[1.0], 3, None).await.unwrap();

        assert_eq!(examples.len(), 1);
        assert!(
            examples[0].answer.len() <= 500,
            "Answer should be truncated to 500 chars, got {}",
            examples[0].answer.len()
        );
    }

    #[tokio::test]
    async fn get_few_shot_examples_returns_at_most_k() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().times(1).returning(|_, _, _, _| {
            Ok(vec![
                make_search_result("c1", "one", 0.1, "a.md"),
                make_search_result("c2", "two", 0.2, "b.md"),
                make_search_result("c3", "three", 0.3, "c.md"),
                make_search_result("c4", "four", 0.4, "d.md"),
                make_search_result("c5", "five", 0.5, "e.md"),
            ])
        });

        let svc = make_service(mock);
        let examples = svc.get_few_shot_examples(&[1.0], 2, None).await.unwrap();

        assert!(examples.len() <= 2, "Should return at most k=2, got {}", examples.len());
    }

    #[tokio::test]
    async fn get_few_shot_examples_fetches_k_times_3() {
        let mut mock = MockVectorStore::new();
        mock.expect_search()
            .times(1)
            .withf(|_, k, _, _| *k == 9) // k=3 -> fetches 3*3=9
            .returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let _ = svc.get_few_shot_examples(&[1.0], 3, None).await.unwrap();
    }

    #[tokio::test]
    async fn get_few_shot_examples_maps_metadata() {
        let mut mock = MockVectorStore::new();
        mock.expect_search().returning(|_, _, _, _| {
            Ok(vec![make_search_result("c1", "content", 0.1, "f.md")])
        });

        let svc = make_service(mock);
        let examples = svc.get_few_shot_examples(&[1.0], 3, None).await.unwrap();

        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].topic, "rust");
        assert_eq!(examples[0].difficulty, "medium");
        assert_eq!(examples[0].source_file, "f.md");
    }

    #[tokio::test]
    async fn get_few_shot_examples_passes_topic_filter() {
        let mut mock = MockVectorStore::new();
        mock.expect_search()
            .times(1)
            .withf(|_, _, filter: &Option<MetadataFilter>, _| {
                filter
                    .as_ref()
                    .map_or(false, |f| f.field == "topic" && f.value == "python")
            })
            .returning(|_, _, _, _| Ok(vec![]));

        let svc = make_service(mock);
        let _ = svc
            .get_few_shot_examples(&[1.0], 3, Some("python"))
            .await
            .unwrap();
    }

    // ===== Send + Sync =====

    #[test]
    fn service_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RelatedConcept>();
        assert_send_sync::<DuplicateCheckResult>();
        assert_send_sync::<FewShotExample>();
        assert_send_sync::<RagService<MockVectorStore>>();
    }
}
