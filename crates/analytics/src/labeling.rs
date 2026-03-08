use std::collections::HashMap;

use serde::Serialize;

use crate::AnalyticsError;

/// A topic assignment for a note.
#[derive(Debug, Clone, Serialize)]
pub struct TopicAssignment {
    pub note_id: i64,
    pub topic_id: i64,
    pub topic_path: String,
    pub confidence: f64,
    /// Labeling method (e.g. "embedding").
    pub method: String,
}

/// Statistics from a labeling operation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LabelingStats {
    pub notes_processed: usize,
    pub assignments_created: usize,
    pub topics_matched: usize,
}

/// Cosine similarity between two equal-length f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Topic labeler. Generic over embedding provider.
pub struct TopicLabeler<E: indexer::embeddings::EmbeddingProvider> {
    pub embedding: E,
    pub db: sqlx::PgPool,
}

impl<E: indexer::embeddings::EmbeddingProvider> TopicLabeler<E> {
    pub fn new(embedding: E, db: sqlx::PgPool) -> Self {
        Self { embedding, db }
    }

    /// Embed all topic descriptions/labels. Returns path -> embedding vector.
    pub async fn embed_topics(
        &self,
        _taxonomy: &super::taxonomy::Taxonomy,
    ) -> Result<HashMap<String, Vec<f32>>, AnalyticsError> {
        todo!()
    }

    /// Label all notes in database with matching topics.
    pub async fn label_notes(
        &self,
        _taxonomy: &super::taxonomy::Taxonomy,
        _min_confidence: f32,
        _max_topics_per_note: usize,
        _batch_size: usize,
    ) -> Result<LabelingStats, AnalyticsError> {
        todo!()
    }

    /// Label a single note.
    pub async fn label_single_note(
        &self,
        _note_id: i64,
        _taxonomy: &super::taxonomy::Taxonomy,
        _topic_embeddings: Option<&HashMap<String, Vec<f32>>>,
        _min_confidence: f32,
        _max_topics: usize,
    ) -> Result<Vec<TopicAssignment>, AnalyticsError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5, "expected ~-1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_known_value() {
        // cos(45 degrees) = sqrt(2)/2 ~ 0.7071
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (sim - expected).abs() < 1e-5,
            "expected ~{expected}, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_unit_vectors() {
        let a = vec![0.6_f32, 0.8]; // unit vector
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }
}
