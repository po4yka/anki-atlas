use std::collections::BTreeMap;
use std::str::FromStr;

use anyhow::{Context, Result};
use common::config::Settings;
use database::run_migrations;
use indexer::embeddings::{EmbeddingProvider, MockEmbeddingProvider, content_hash};
use indexer::qdrant::{NotePayload, VectorRepository};
use qdrant_client::Qdrant;
use rustis::commands::{FlushingMode, ServerCommands};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use surface_runtime::workflows::QdrantVectorStore;

pub const PERF_COLLECTION_NAME: &str = "anki_notes";

const SEARCH_TERMS: &[&str] = &[
    "ownership",
    "borrowing",
    "lifetimes",
    "tokio",
    "sqlx",
    "qdrant",
    "redis",
    "async",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetProfile {
    Pr,
    Nightly,
}

impl DatasetProfile {
    pub fn spec(self) -> DatasetSpec {
        match self {
            Self::Pr => DatasetSpec {
                notes: 1_000,
                topics: 120,
                duplicate_clusters: 25,
            },
            Self::Nightly => DatasetSpec {
                notes: 10_000,
                topics: 1_000,
                duplicate_clusters: 250,
            },
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pr => "pr",
            Self::Nightly => "nightly",
        }
    }
}

impl FromStr for DatasetProfile {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "pr" => Ok(Self::Pr),
            "nightly" => Ok(Self::Nightly),
            other => anyhow::bail!("unsupported dataset profile: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetSpec {
    pub notes: usize,
    pub topics: usize,
    pub duplicate_clusters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedManifest {
    pub profile: DatasetProfile,
    pub notes: usize,
    pub topics: usize,
    pub duplicate_clusters: usize,
    pub search_queries: Vec<String>,
    pub root_topics: Vec<String>,
    pub branch_topics: Vec<String>,
    pub leaf_topics: Vec<String>,
    pub duplicate_deck: String,
    pub duplicate_tag: String,
    pub sync_source: String,
}

#[derive(Debug, Clone)]
struct TopicSeed {
    path: String,
    label: String,
    description: Option<String>,
}

#[derive(Debug, Clone)]
struct TopicHandle {
    topic_id: i32,
    path: String,
}

#[derive(Debug, Clone)]
struct SeededNote {
    note_id: i64,
    model_id: i64,
    deck_name: String,
    tags: Vec<String>,
    normalized_text: String,
    lapses: i32,
    reps: i32,
    fail_rate: Option<f64>,
}

#[derive(Debug, Clone)]
struct SeedState {
    manifest: SeedManifest,
    notes: Vec<SeededNote>,
}

pub async fn reset_and_seed(settings: &Settings, profile: DatasetProfile) -> Result<SeedManifest> {
    let pool = database::create_pool(&settings.database())
        .await
        .context("create PostgreSQL pool for perf seeding")?;
    run_migrations(&pool)
        .await
        .context("run migrations for perf seeding")?;
    truncate_runtime_tables(&pool).await?;
    flush_redis(&settings.redis_url).await?;

    let state = seed_postgres_internal(&pool, profile).await?;
    reset_qdrant(settings, &state.notes).await?;
    Ok(state.manifest)
}

pub async fn seed_postgres_only(pool: &PgPool, profile: DatasetProfile) -> Result<SeedManifest> {
    run_migrations(pool)
        .await
        .context("run migrations for Postgres bench fixture")?;
    truncate_runtime_tables(pool).await?;
    Ok(seed_postgres_internal(pool, profile).await?.manifest)
}

pub fn profile_manifest(profile: DatasetProfile) -> SeedManifest {
    let spec = profile.spec();
    let topics = build_topics(spec);
    let root_topics = topics
        .iter()
        .filter(|topic| !topic.path.contains('/'))
        .take(4)
        .map(|topic| topic.path.clone())
        .collect::<Vec<_>>();
    let branch_topics = topics
        .iter()
        .filter(|topic| topic.path.matches('/').count() == 1)
        .take(4)
        .map(|topic| topic.path.clone())
        .collect::<Vec<_>>();
    let leaf_topics = topics
        .iter()
        .filter(|topic| topic.path.matches('/').count() == 2)
        .take(6)
        .map(|topic| topic.path.clone())
        .collect::<Vec<_>>();

    SeedManifest {
        profile,
        notes: spec.notes,
        topics: spec.topics,
        duplicate_clusters: spec.duplicate_clusters,
        search_queries: SEARCH_TERMS
            .iter()
            .map(|term| (*term).to_string())
            .collect(),
        root_topics,
        branch_topics,
        leaf_topics,
        duplicate_deck: "Deck 00".to_string(),
        duplicate_tag: "dup-cluster-000".to_string(),
        sync_source: "/tmp/perf-source.anki2".to_string(),
    }
}

async fn flush_redis(redis_url: &str) -> Result<()> {
    let client = jobs::connection::create_redis_client(redis_url)
        .await
        .context("connect Redis for perf reset")?;
    client
        .flushdb(FlushingMode::Sync)
        .await
        .context("flush Redis for perf reset")?;
    Ok(())
}

async fn truncate_runtime_tables(pool: &PgPool) -> Result<()> {
    sqlx::query(
        "TRUNCATE TABLE \
            note_topics, \
            topics, \
            card_stats, \
            cards, \
            decks, \
            notes, \
            models, \
            sync_metadata \
         RESTART IDENTITY CASCADE",
    )
    .execute(pool)
    .await
    .context("truncate runtime tables for perf reset")?;

    sqlx::query(
        "INSERT INTO sync_metadata (key, value) VALUES \
            ('schema_version', '\"002\"'), \
            ('normalization_version', '\"1\"'), \
            ('last_sync_at', 'null') \
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
    )
    .execute(pool)
    .await
    .context("rebuild sync metadata after perf reset")?;

    Ok(())
}

async fn reset_qdrant(settings: &Settings, notes: &[SeededNote]) -> Result<()> {
    let client = Qdrant::from_url(&settings.qdrant_url)
        .build()
        .context("connect Qdrant for perf seeding")?;
    let _ = client.delete_collection(PERF_COLLECTION_NAME).await;

    let vector_store = QdrantVectorStore::new(client, PERF_COLLECTION_NAME);
    vector_store
        .ensure_collection(settings.embedding_dimension as usize)
        .await
        .context("ensure Qdrant collection for perf seeding")?;

    let embedding = MockEmbeddingProvider::new(settings.embedding_dimension as usize);
    let texts = notes
        .iter()
        .map(|note| note.normalized_text.clone())
        .collect::<Vec<_>>();
    let vectors = embedding
        .embed(&texts)
        .await
        .context("embed perf notes for Qdrant")?;
    let payloads = notes
        .iter()
        .map(|note| NotePayload {
            note_id: note.note_id,
            deck_names: vec![note.deck_name.clone()],
            tags: note.tags.clone(),
            model_id: note.model_id,
            content_hash: content_hash(embedding.model_name(), &note.normalized_text),
            mature: note.reps >= 21,
            lapses: note.lapses,
            reps: note.reps,
            fail_rate: note.fail_rate,
        })
        .collect::<Vec<_>>();

    for (vector_chunk, payload_chunk) in vectors.chunks(256).zip(payloads.chunks(256)) {
        vector_store
            .upsert_vectors(vector_chunk, payload_chunk, None)
            .await
            .context("upsert Qdrant perf vectors")?;
    }

    Ok(())
}

async fn seed_postgres_internal(pool: &PgPool, profile: DatasetProfile) -> Result<SeedState> {
    let spec = profile.spec();
    let topics = build_topics(spec);
    let mut txn = pool.begin().await.context("begin perf seed transaction")?;

    let deck_count = 12_i64;
    for deck_id in 0..deck_count {
        let name = format!("Deck {:02}", deck_id);
        sqlx::query("INSERT INTO decks (deck_id, name, parent_name, config) VALUES ($1, $2, NULL, '{}'::jsonb)")
            .bind(100_i64 + deck_id)
            .bind(&name)
            .execute(&mut *txn)
            .await
            .with_context(|| format!("insert deck {name}"))?;
    }

    let mut topic_handles = Vec::with_capacity(topics.len());
    for topic in &topics {
        let row: (i32,) = sqlx::query_as(
            "INSERT INTO topics (path, label, description) VALUES ($1, $2, $3) RETURNING topic_id",
        )
        .bind(&topic.path)
        .bind(&topic.label)
        .bind(&topic.description)
        .fetch_one(&mut *txn)
        .await
        .with_context(|| format!("insert topic {}", topic.path))?;

        topic_handles.push(TopicHandle {
            topic_id: row.0,
            path: topic.path.clone(),
        });
    }

    let roots = topic_handles
        .iter()
        .filter(|topic| !topic.path.contains('/'))
        .cloned()
        .collect::<Vec<_>>();
    let branches = topic_handles
        .iter()
        .filter(|topic| topic.path.matches('/').count() == 1)
        .cloned()
        .collect::<Vec<_>>();
    let leaves = topic_handles
        .iter()
        .filter(|topic| topic.path.matches('/').count() == 2)
        .cloned()
        .collect::<Vec<_>>();

    let sparse_root_index = roots.len().saturating_sub(1);
    let weak_root_index = usize::min(1, roots.len().saturating_sub(1));
    let duplicate_notes = spec.duplicate_clusters * 3;
    let mut notes = Vec::with_capacity(spec.notes);

    for idx in 0..spec.notes {
        let note_id = (idx + 1) as i64;
        let model_id = 200_i64 + (idx % 4) as i64;
        let root_index = idx % sparse_root_index.max(1);
        let root = &roots[root_index];
        let branch = &branches[idx % branches.len()];
        let leaf = &leaves[idx % leaves.len()];
        let concept = SEARCH_TERMS[idx % SEARCH_TERMS.len()];
        let deck_id = 100_i64 + (root_index % deck_count as usize) as i64;
        let deck_name = format!("Deck {:02}", deck_id - 100);
        let duplicate_cluster = idx / 3;

        let normalized_text = if idx < duplicate_notes {
            format!(
                "duplicate cluster {duplicate_cluster:03} {concept} {}",
                leaf.path
            )
        } else {
            format!(
                "{concept} practice note {note_id:05} {} {}",
                branch.path, leaf.path
            )
        };

        let mut tags = vec![
            concept.to_string(),
            format!("root-{root_index:03}"),
            format!("tag-{:02}", idx % 24),
        ];
        if idx < duplicate_notes {
            tags.push(format!("dup-cluster-{duplicate_cluster:03}"));
        }

        let reps = if root_index == weak_root_index {
            18 + (idx % 12) as i32
        } else {
            4 + (idx % 10) as i32
        };
        let lapses = if root_index == weak_root_index && idx % 4 == 0 {
            6 + (idx % 4) as i32
        } else {
            (idx % 3) as i32
        };
        let fail_rate = if root_index == weak_root_index && idx % 4 == 0 {
            Some(0.25)
        } else if idx % 11 == 0 {
            Some(0.12)
        } else {
            Some(0.03)
        };

        sqlx::query(
            "INSERT INTO notes \
             (note_id, model_id, tags, fields_json, raw_fields, normalized_text, mtime, usn) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(note_id)
        .bind(model_id)
        .bind(&tags)
        .bind(serde_json::json!({
            "Front": format!("What about {concept}?"),
            "Back": normalized_text,
        }))
        .bind(Some("Front\x1fBack"))
        .bind(&normalized_text)
        .bind(note_id)
        .bind(0_i32)
        .execute(&mut *txn)
        .await
        .with_context(|| format!("insert note {note_id}"))?;

        let card_id = 10_000_i64 + note_id;
        sqlx::query(
            "INSERT INTO cards \
             (card_id, note_id, deck_id, ord, due, ivl, ease, lapses, reps, queue, type, mtime, usn) \
             VALUES ($1, $2, $3, 0, 0, $4, 2500, $5, $6, 0, 0, $7, 0)",
        )
        .bind(card_id)
        .bind(note_id)
        .bind(deck_id)
        .bind(reps)
        .bind(lapses)
        .bind(reps)
        .bind(note_id)
        .execute(&mut *txn)
        .await
        .with_context(|| format!("insert card for note {note_id}"))?;

        sqlx::query(
            "INSERT INTO card_stats \
             (card_id, reviews, avg_ease, fail_rate, last_review_at, total_time_ms) \
             VALUES ($1, $2, 2.5, $3, NOW(), $4)",
        )
        .bind(card_id)
        .bind(reps)
        .bind(fail_rate)
        .bind(i64::from(reps) * 850_i64)
        .execute(&mut *txn)
        .await
        .with_context(|| format!("insert card stats for note {note_id}"))?;

        let confidence = 0.68_f32 + (idx % 5) as f32 * 0.04_f32;
        sqlx::query(
            "INSERT INTO note_topics (note_id, topic_id, confidence, method) VALUES ($1, $2, $3, $4)",
        )
        .bind(note_id)
        .bind(leaf.topic_id)
        .bind(confidence)
        .bind("perf_seed")
        .execute(&mut *txn)
        .await
        .with_context(|| format!("assign leaf topic to note {note_id}"))?;

        if idx % 5 == 0 {
            sqlx::query(
                "INSERT INTO note_topics (note_id, topic_id, confidence, method) VALUES ($1, $2, $3, $4)",
            )
            .bind(note_id)
            .bind(root.topic_id)
            .bind((confidence - 0.08_f32).max(0.5_f32))
            .bind("perf_seed")
            .execute(&mut *txn)
            .await
            .with_context(|| format!("assign root topic to note {note_id}"))?;
        }

        notes.push(SeededNote {
            note_id,
            model_id,
            deck_name,
            tags,
            normalized_text,
            lapses,
            reps,
            fail_rate,
        });
    }

    if let Some(sparse_root) = roots.get(sparse_root_index) {
        for leaf in leaves.iter().rev().take(3) {
            let synthetic_note_id = 100_000_i64 + i64::from(leaf.topic_id);
            sqlx::query(
                "INSERT INTO notes \
                 (note_id, model_id, tags, fields_json, raw_fields, normalized_text, mtime, usn) \
                 VALUES ($1, 999, $2, '{}'::jsonb, NULL, $3, $4, 0)",
            )
            .bind(synthetic_note_id)
            .bind(vec!["gap".to_string()])
            .bind(format!("sparse topic {}", leaf.path))
            .bind(synthetic_note_id)
            .execute(&mut *txn)
            .await
            .with_context(|| format!("insert sparse note for {}", leaf.path))?;

            sqlx::query(
                "INSERT INTO note_topics (note_id, topic_id, confidence, method) VALUES ($1, $2, 0.55, $3)",
            )
            .bind(synthetic_note_id)
            .bind(sparse_root.topic_id)
            .bind("perf_seed")
            .execute(&mut *txn)
            .await
            .context("assign sparse root topic")?;
        }
    }

    txn.commit().await.context("commit perf seed transaction")?;

    Ok(SeedState {
        manifest: profile_manifest(profile),
        notes,
    })
}

fn build_topics(spec: DatasetSpec) -> Vec<TopicSeed> {
    let root_count = usize::min(12, usize::max(6, spec.topics / 10));
    let branch_count = usize::max(root_count * 2, spec.topics / 4);
    let leaf_count = spec.topics.saturating_sub(root_count + branch_count);
    let mut topics = Vec::with_capacity(spec.topics);

    for root_index in 0..root_count {
        let path = format!("root-{root_index:03}");
        topics.push(TopicSeed {
            path: path.clone(),
            label: format!("Root {root_index:03}"),
            description: Some(format!("Synthetic root topic {path}")),
        });
    }

    for branch_index in 0..branch_count {
        let root_index = branch_index % root_count;
        let path = format!("root-{root_index:03}/branch-{branch_index:03}");
        topics.push(TopicSeed {
            path: path.clone(),
            label: format!("Branch {branch_index:03}"),
            description: Some(format!("Synthetic branch topic {path}")),
        });
    }

    for leaf_index in 0..leaf_count {
        let branch_index = leaf_index % branch_count;
        let root_index = branch_index % root_count;
        let path = format!("root-{root_index:03}/branch-{branch_index:03}/leaf-{leaf_index:03}");
        topics.push(TopicSeed {
            path: path.clone(),
            label: format!("Leaf {leaf_index:03}"),
            description: Some(format!("Synthetic leaf topic {path}")),
        });
    }

    topics
}

pub fn manifest_json(manifest: &SeedManifest) -> Result<String> {
    serde_json::to_string_pretty(manifest).context("serialize perf seed manifest")
}

pub fn summarize_requests(
    requests: impl Iterator<Item = (String, usize)>,
) -> BTreeMap<String, usize> {
    let mut summary = BTreeMap::new();
    for (name, count) in requests {
        *summary.entry(name).or_insert(0) += count;
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_profiles_match_planned_sizes() {
        assert_eq!(
            DatasetProfile::Pr.spec(),
            DatasetSpec {
                notes: 1_000,
                topics: 120,
                duplicate_clusters: 25,
            }
        );
        assert_eq!(
            DatasetProfile::Nightly.spec(),
            DatasetSpec {
                notes: 10_000,
                topics: 1_000,
                duplicate_clusters: 250,
            }
        );
    }

    #[test]
    fn generated_topics_cover_requested_count() {
        let topics = build_topics(DatasetProfile::Nightly.spec());
        assert_eq!(topics.len(), DatasetProfile::Nightly.spec().topics);
        assert!(topics[0].path.starts_with("root-"));
        assert!(topics.last().unwrap().path.contains("/leaf-"));
    }

    #[test]
    fn profile_parsing_is_stable() {
        assert_eq!(DatasetProfile::from_str("pr").unwrap(), DatasetProfile::Pr);
        assert_eq!(
            DatasetProfile::from_str("nightly").unwrap(),
            DatasetProfile::Nightly
        );
    }
}
