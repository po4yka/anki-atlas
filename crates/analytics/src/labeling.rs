use std::collections::HashMap;

use serde::Serialize;

use crate::AnalyticsError;

/// Method used to assign a topic to a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelingMethod {
    Embedding,
}

impl LabelingMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
        }
    }
}

/// A topic assignment for a note.
#[derive(Debug, Clone, Serialize)]
pub struct TopicAssignment {
    pub note_id: i64,
    pub topic_id: i64,
    pub topic_path: String,
    pub confidence: f64,
    pub method: LabelingMethod,
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
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Rank topics by cosine similarity to a note embedding.
/// Returns assignments above `min_confidence`, limited to `max_topics`,
/// sorted by confidence descending.
#[allow(dead_code)]
pub(crate) fn rank_topics_for_note(
    note_id: i64,
    note_embedding: &[f32],
    topic_embeddings: &HashMap<String, (i64, Vec<f32>)>,
    min_confidence: f32,
    max_topics: usize,
) -> Vec<TopicAssignment> {
    let mut assignments: Vec<TopicAssignment> = topic_embeddings
        .iter()
        .filter_map(|(path, (topic_id, emb))| {
            let sim = cosine_similarity(note_embedding, emb);
            if sim >= min_confidence {
                Some(TopicAssignment {
                    note_id,
                    topic_id: *topic_id,
                    topic_path: path.clone(),
                    confidence: f64::from(sim),
                    method: LabelingMethod::Embedding,
                })
            } else {
                None
            }
        })
        .collect();
    assignments.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assignments.truncate(max_topics);
    assignments
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
        taxonomy: &super::taxonomy::Taxonomy,
    ) -> Result<HashMap<String, Vec<f32>>, AnalyticsError> {
        let topic_paths: Vec<String> = taxonomy.topics.keys().cloned().collect();
        let topic_texts: Vec<String> = topic_paths
            .iter()
            .map(|path| {
                let topic = &taxonomy.topics[path];
                match &topic.description {
                    Some(desc) => format!("{}: {desc}", topic.label),
                    None => topic.label.clone(),
                }
            })
            .collect();

        let embeddings = self.embedding.embed(&topic_texts).await?;

        let result: HashMap<String, Vec<f32>> = topic_paths.into_iter().zip(embeddings).collect();

        Ok(result)
    }

    fn build_topic_embedding_index(
        taxonomy: &super::taxonomy::Taxonomy,
        topic_embeddings: &HashMap<String, Vec<f32>>,
    ) -> HashMap<String, (i64, Vec<f32>)> {
        topic_embeddings
            .iter()
            .filter_map(|(topic_path, embedding)| {
                taxonomy.topics.get(topic_path).and_then(|topic| {
                    topic
                        .topic_id
                        .map(|topic_id| (topic_path.clone(), (topic_id, embedding.clone())))
                })
            })
            .collect()
    }

    /// Label all notes in database with matching topics.
    pub async fn label_notes(
        &self,
        taxonomy: &super::taxonomy::Taxonomy,
        min_confidence: f32,
        max_topics_per_note: usize,
        batch_size: usize,
    ) -> Result<LabelingStats, AnalyticsError> {
        let topic_embeddings = self.embed_topics(taxonomy).await?;
        let topic_embedding_index = Self::build_topic_embedding_index(taxonomy, &topic_embeddings);

        let mut stats = LabelingStats::default();
        let mut matched_topics = std::collections::HashSet::new();
        let mut offset: i64 = 0;

        loop {
            let notes: Vec<(i64, String)> = sqlx::query_as(
                "SELECT note_id, normalized_text FROM notes \
                 WHERE deleted_at IS NULL \
                 ORDER BY note_id LIMIT $1 OFFSET $2",
            )
            .bind(batch_size as i64)
            .bind(offset)
            .fetch_all(&self.db)
            .await?;

            if notes.is_empty() {
                break;
            }

            let texts: Vec<String> = notes.iter().map(|(_, text)| text.clone()).collect();
            let note_embeddings = self.embedding.embed(&texts).await?;

            for ((note_id, _), note_emb) in notes.iter().zip(note_embeddings.iter()) {
                let assignments = rank_topics_for_note(
                    *note_id,
                    note_emb,
                    &topic_embedding_index,
                    min_confidence,
                    max_topics_per_note,
                );

                for assignment in &assignments {
                    sqlx::query(
                        "INSERT INTO note_topics (note_id, topic_id, confidence, method) \
                         VALUES ($1, $2, $3, $4) \
                         ON CONFLICT (note_id, topic_id) DO UPDATE \
                         SET confidence = $3, method = $4",
                    )
                    .bind(assignment.note_id)
                    .bind(assignment.topic_id as i32)
                    .bind(assignment.confidence as f32)
                    .bind(assignment.method.as_str())
                    .execute(&self.db)
                    .await?;

                    matched_topics.insert(assignment.topic_path.clone());
                }

                stats.assignments_created += assignments.len();
                stats.notes_processed += 1;
            }

            offset += notes.len() as i64;
        }

        stats.topics_matched = matched_topics.len();
        Ok(stats)
    }

    /// Label a single note.
    pub async fn label_single_note(
        &self,
        note_id: i64,
        taxonomy: &super::taxonomy::Taxonomy,
        topic_embeddings: Option<&HashMap<String, Vec<f32>>>,
        min_confidence: f32,
        max_topics: usize,
    ) -> Result<Vec<TopicAssignment>, AnalyticsError> {
        // Get note text
        let (text,): (String,) =
            sqlx::query_as("SELECT normalized_text FROM notes WHERE note_id = $1")
                .bind(note_id)
                .fetch_one(&self.db)
                .await?;

        // Embed the note
        let note_embeddings = self.embedding.embed(&[text]).await?;
        let note_emb = &note_embeddings[0];

        // Get or compute topic embeddings
        let owned_topic_embeddings;
        let topic_embeddings = match topic_embeddings {
            Some(existing_topic_embeddings) => existing_topic_embeddings,
            None => {
                owned_topic_embeddings = self.embed_topics(taxonomy).await?;
                &owned_topic_embeddings
            }
        };

        let topic_embedding_index = Self::build_topic_embedding_index(taxonomy, topic_embeddings);

        let assignments = rank_topics_for_note(
            note_id,
            note_emb,
            &topic_embedding_index,
            min_confidence,
            max_topics,
        );

        Ok(assignments)
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

    // --- rank_topics_for_note ---

    fn make_topic_embeddings(
        entries: &[(&str, i64, Vec<f32>)],
    ) -> HashMap<String, (i64, Vec<f32>)> {
        entries
            .iter()
            .map(|(path, id, emb)| (path.to_string(), (*id, emb.clone())))
            .collect()
    }

    #[test]
    fn rank_topics_skips_below_min_confidence() {
        // Note embedding is unit vector along x-axis
        let note_emb = vec![1.0_f32, 0.0, 0.0];
        // Topic A: nearly aligned (high similarity ~0.98)
        // Topic B: mostly orthogonal (low similarity ~0.17)
        let topics = make_topic_embeddings(&[
            ("topicA", 1, vec![0.98, 0.2, 0.0]),
            ("topicB", 2, vec![0.1, 0.98, 0.17]),
        ]);
        let result = rank_topics_for_note(42, &note_emb, &topics, 0.8, 10);
        // Only topicA should pass the 0.8 threshold
        assert_eq!(result.len(), 1, "should skip topics below min_confidence");
        assert_eq!(result[0].topic_path, "topicA");
        assert!(result[0].confidence >= 0.8);
    }

    #[test]
    fn rank_topics_respects_max_topics_limit() {
        let note_emb = vec![1.0_f32, 0.0, 0.0];
        // All topics have high similarity
        let topics = make_topic_embeddings(&[
            ("a", 1, vec![1.0, 0.0, 0.0]),
            ("b", 2, vec![0.99, 0.14, 0.0]),
            ("c", 3, vec![0.98, 0.2, 0.0]),
            ("d", 4, vec![0.97, 0.24, 0.0]),
        ]);
        let result = rank_topics_for_note(42, &note_emb, &topics, 0.0, 2);
        assert_eq!(result.len(), 2, "should limit to max_topics=2");
    }

    #[test]
    fn rank_topics_sorted_by_confidence_descending() {
        let note_emb = vec![1.0_f32, 0.0, 0.0];
        let topics = make_topic_embeddings(&[
            ("low", 1, vec![0.7, 0.7, 0.0]),
            ("high", 2, vec![1.0, 0.0, 0.0]),
            ("mid", 3, vec![0.9, 0.4, 0.0]),
        ]);
        let result = rank_topics_for_note(42, &note_emb, &topics, 0.0, 10);
        assert!(result.len() >= 2);
        for w in result.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "expected descending order: {} >= {}",
                w[0].confidence,
                w[1].confidence
            );
        }
    }

    #[test]
    fn rank_topics_empty_topic_embeddings() {
        let note_emb = vec![1.0_f32, 0.0];
        let topics = HashMap::new();
        let result = rank_topics_for_note(42, &note_emb, &topics, 0.0, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn rank_topics_sets_method_to_embedding() {
        let note_emb = vec![1.0_f32, 0.0];
        let topics = make_topic_embeddings(&[("t", 1, vec![1.0, 0.0])]);
        let result = rank_topics_for_note(42, &note_emb, &topics, 0.0, 10);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].method, LabelingMethod::Embedding);
        assert_eq!(result[0].note_id, 42);
        assert_eq!(result[0].topic_id, 1);
    }
}
