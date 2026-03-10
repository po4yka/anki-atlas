use crate::AnalyticsError;
use crate::coverage::{TopicCoverage, TopicGap, WeakNote};
use crate::duplicates::{DuplicateCluster, DuplicateDetail, DuplicateStats};
use crate::labeling::LabelingStats;
use crate::taxonomy::Taxonomy;

/// Facade aggregating taxonomy, coverage, labeling, and duplicate detection.
pub struct AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub embedding: E,
    pub vector_repo: V,
    pub db: sqlx::PgPool,
}

impl<E, V> AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub fn new(embedding: E, vector_repo: V, db: sqlx::PgPool) -> Self {
        Self {
            embedding,
            vector_repo,
            db,
        }
    }

    /// Load taxonomy from YAML (syncing to DB) or from DB.
    pub async fn load_taxonomy(
        &self,
        yaml_path: Option<&std::path::Path>,
    ) -> Result<Taxonomy, AnalyticsError> {
        match yaml_path {
            Some(path) => {
                let mut taxonomy = crate::taxonomy::load_taxonomy_from_yaml(path)?;
                if !taxonomy.topics.is_empty() {
                    let id_map = crate::taxonomy::sync_taxonomy_to_db(&self.db, &taxonomy).await?;
                    taxonomy.apply_topic_ids(&id_map);
                }
                Ok(taxonomy)
            }
            None => crate::taxonomy::load_taxonomy_from_db(&self.db).await,
        }
    }

    /// Label all notes with topics.
    pub async fn label_notes(
        &self,
        taxonomy: Option<&Taxonomy>,
        min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError> {
        let owned_taxonomy;
        let tax = match taxonomy {
            Some(t) => t,
            None => {
                owned_taxonomy = crate::taxonomy::load_taxonomy_from_db(&self.db).await?;
                &owned_taxonomy
            }
        };

        // Embed topics
        let paths: Vec<String> = tax.topics.keys().cloned().collect();
        let texts: Vec<String> = paths
            .iter()
            .map(|p| {
                let topic = &tax.topics[p];
                match &topic.description {
                    Some(desc) => format!("{}: {desc}", topic.label),
                    None => topic.label.clone(),
                }
            })
            .collect();
        let topic_vecs = self.embedding.embed(&texts).await?;
        let topic_emb_with_ids: std::collections::HashMap<String, (i64, Vec<f32>)> = paths
            .into_iter()
            .zip(topic_vecs)
            .filter_map(|(path, emb)| {
                tax.topics
                    .get(&path)
                    .and_then(|t| t.topic_id.map(|id| (path, (id, emb))))
            })
            .collect();

        let max_topics_per_note = 5;
        let batch_size: i64 = 100;
        let mut stats = LabelingStats::default();
        let mut matched_topics = std::collections::HashSet::new();
        let mut offset: i64 = 0;

        loop {
            let notes: Vec<(i64, String)> = sqlx::query_as(
                "SELECT note_id, normalized_text FROM notes \
                 WHERE deleted_at IS NULL ORDER BY note_id LIMIT $1 OFFSET $2",
            )
            .bind(batch_size)
            .bind(offset)
            .fetch_all(&self.db)
            .await?;

            if notes.is_empty() {
                break;
            }

            let note_texts: Vec<String> = notes.iter().map(|(_, t)| t.clone()).collect();
            let note_vecs = self.embedding.embed(&note_texts).await?;

            for ((note_id, _), note_emb) in notes.iter().zip(note_vecs.iter()) {
                let assignments = crate::labeling::rank_topics_for_note(
                    *note_id,
                    note_emb,
                    &topic_emb_with_ids,
                    min_confidence,
                    max_topics_per_note,
                );
                for a in &assignments {
                    sqlx::query(
                        "INSERT INTO note_topics (note_id, topic_id, confidence, method) \
                         VALUES ($1, $2, $3, $4) \
                         ON CONFLICT (note_id, topic_id) DO UPDATE SET confidence = $3, method = $4",
                    )
                    .bind(a.note_id)
                    .bind(a.topic_id as i32)
                    .bind(a.confidence as f32)
                    .bind(a.method.as_str())
                    .execute(&self.db)
                    .await?;
                    matched_topics.insert(a.topic_path.clone());
                }
                stats.assignments_created += assignments.len();
                stats.notes_processed += 1;
            }

            offset += notes.len() as i64;
        }

        stats.topics_matched = matched_topics.len();
        Ok(stats)
    }

    /// Get coverage metrics for a topic.
    pub async fn get_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        crate::coverage::get_topic_coverage(&self.db, topic_path, include_subtree).await
    }

    /// Find gaps in topic coverage under a root path.
    pub async fn get_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        crate::coverage::get_topic_gaps(&self.db, topic_path, min_coverage).await
    }

    /// Get weak notes (high lapse rate) in a topic subtree.
    pub async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        crate::coverage::get_weak_notes(&self.db, topic_path, max_results, 0.1).await
    }

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        // Inline: fetch notes, find similar via vector_repo, cluster with UnionFind
        let note_ids: Vec<(i64,)> =
            sqlx::query_as("SELECT note_id FROM notes WHERE deleted_at IS NULL ORDER BY note_id")
                .fetch_all(&self.db)
                .await?;

        let mut uf = crate::duplicates::UnionFind::new();
        let mut pair_scores: std::collections::HashMap<(i64, i64), f64> =
            std::collections::HashMap::new();

        for (note_id,) in &note_ids {
            let similar = self
                .vector_repo
                .find_similar_to_note(*note_id, 20, threshold as f32, deck_filter, tag_filter)
                .await?;
            for (other_id, score) in similar {
                if other_id == *note_id {
                    continue;
                }
                let pair = if *note_id < other_id {
                    (*note_id, other_id)
                } else {
                    (other_id, *note_id)
                };
                pair_scores.entry(pair).or_insert(f64::from(score));
                uf.union(*note_id, other_id);
            }
        }

        let mut cluster_members: std::collections::HashMap<i64, Vec<i64>> =
            std::collections::HashMap::new();
        for (note_id,) in &note_ids {
            let root = uf.find(*note_id);
            cluster_members.entry(root).or_default().push(*note_id);
        }

        let mut clusters = Vec::new();
        for members in cluster_members.values() {
            if members.len() < 2 {
                continue;
            }
            let mut best_id = members[0];
            let mut best_reviews: i64 = 0;
            for &nid in members {
                let (reviews,): (i64,) = sqlx::query_as(
                    "SELECT COALESCE(SUM(c.reps), 0) FROM cards c WHERE c.note_id = $1",
                )
                .bind(nid)
                .fetch_one(&self.db)
                .await?;
                if reviews > best_reviews {
                    best_reviews = reviews;
                    best_id = nid;
                }
            }

            let (rep_text,): (String,) =
                sqlx::query_as("SELECT LEFT(normalized_text, 200) FROM notes WHERE note_id = $1")
                    .bind(best_id)
                    .fetch_one(&self.db)
                    .await?;

            let mut duplicates = Vec::new();
            let mut all_decks = Vec::new();
            let mut all_tags = Vec::new();

            for &nid in members {
                if nid == best_id {
                    continue;
                }
                let pair = if best_id < nid {
                    (best_id, nid)
                } else {
                    (nid, best_id)
                };
                let sim = pair_scores.get(&pair).copied().unwrap_or(0.0);
                let detail: (String, Vec<String>) = sqlx::query_as(
                    "SELECT LEFT(n.normalized_text, 200), COALESCE(n.tags, '{}') \
                     FROM notes n WHERE n.note_id = $1",
                )
                .bind(nid)
                .fetch_one(&self.db)
                .await?;
                let dn: Vec<String> = sqlx::query_as::<_, (String,)>(
                    "SELECT DISTINCT d.name FROM cards c \
                     JOIN decks d ON d.deck_id = c.deck_id WHERE c.note_id = $1",
                )
                .bind(nid)
                .fetch_all(&self.db)
                .await?
                .into_iter()
                .map(|(n,)| n)
                .collect();

                all_decks.extend(dn.clone());
                all_tags.extend(detail.1.clone());
                duplicates.push(DuplicateDetail {
                    note_id: nid,
                    similarity: sim,
                    text: detail.0,
                    deck_names: dn,
                    tags: detail.1,
                });
            }

            let rep_dn: Vec<String> = sqlx::query_as::<_, (String,)>(
                "SELECT DISTINCT d.name FROM cards c \
                 JOIN decks d ON d.deck_id = c.deck_id WHERE c.note_id = $1",
            )
            .bind(best_id)
            .fetch_all(&self.db)
            .await?
            .into_iter()
            .map(|(n,)| n)
            .collect();
            let (rep_tags,): (Vec<String>,) =
                sqlx::query_as("SELECT COALESCE(tags, '{}') FROM notes WHERE note_id = $1")
                    .bind(best_id)
                    .fetch_one(&self.db)
                    .await?;

            all_decks.extend(rep_dn);
            all_tags.extend(rep_tags);
            all_decks.sort();
            all_decks.dedup();
            all_tags.sort();
            all_tags.dedup();

            clusters.push(DuplicateCluster {
                representative_id: best_id,
                representative_text: rep_text,
                duplicates,
                deck_names: all_decks,
                tags: all_tags,
            });
        }

        clusters.sort_by_key(|b| std::cmp::Reverse(b.size()));
        clusters.truncate(max_clusters);

        let total_duplicates: usize = clusters.iter().map(|c| c.duplicates.len()).sum();
        let clusters_found = clusters.len();
        let avg_cluster_size = if clusters.is_empty() {
            0.0
        } else {
            clusters.iter().map(|c| c.size()).sum::<usize>() as f64 / clusters_found as f64
        };

        Ok((
            clusters,
            DuplicateStats {
                notes_scanned: note_ids.len(),
                clusters_found,
                total_duplicates,
                avg_cluster_size,
            },
        ))
    }

    /// Get coverage tree for all topics (optionally filtered by root path).
    pub async fn get_taxonomy_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        crate::coverage::get_coverage_tree(&self.db, root_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all public types are Send + Sync.
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn analytics_error_is_send_sync() {
        assert_send_sync::<AnalyticsError>();
    }

    #[test]
    fn topic_is_send_sync() {
        assert_send_sync::<crate::taxonomy::Topic>();
    }

    #[test]
    fn taxonomy_is_send_sync() {
        assert_send_sync::<crate::taxonomy::Taxonomy>();
    }

    #[test]
    fn topic_coverage_is_send_sync() {
        assert_send_sync::<TopicCoverage>();
    }

    #[test]
    fn topic_gap_is_send_sync() {
        assert_send_sync::<TopicGap>();
    }

    #[test]
    fn weak_note_is_send_sync() {
        assert_send_sync::<WeakNote>();
    }

    #[test]
    fn labeling_stats_is_send_sync() {
        assert_send_sync::<LabelingStats>();
    }

    #[test]
    fn duplicate_cluster_is_send_sync() {
        assert_send_sync::<DuplicateCluster>();
    }

    #[test]
    fn duplicate_stats_is_send_sync() {
        assert_send_sync::<DuplicateStats>();
    }

    #[test]
    fn analytics_service_is_generic() {
        // Verify the service compiles with mock types
        // (this is a compile-time check, the test body doesn't matter)
        fn _check<
            E: indexer::embeddings::EmbeddingProvider,
            V: indexer::qdrant::VectorRepository,
        >() {
            assert_send_sync::<AnalyticsService<E, V>>();
        }
    }
}
