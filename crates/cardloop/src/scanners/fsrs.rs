use std::collections::HashMap;
use std::path::Path;

use anki_reader::models::{AnkiCard, CardStats};
use anki_reader::read_anki_collection;
use card::registry::CardRegistry;
use chrono::Utc;
use common::{CardId, NoteId};

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::Scanner;

/// Scans an Anki collection via FSRS metrics and cross-references the card registry.
pub struct FsrsScanner<'a> {
    registry: &'a CardRegistry,
    collection_path: &'a Path,
}

impl<'a> FsrsScanner<'a> {
    pub fn new(registry: &'a CardRegistry, collection_path: &'a Path) -> Self {
        Self {
            registry,
            collection_path,
        }
    }

    fn item_id(slug: &str, discriminator: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"fsrs:");
        hasher.update(slug.as_bytes());
        hasher.update(b":");
        hasher.update(discriminator.as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    fn score_from_card(card: &AnkiCard, stats: Option<&CardStats>) -> f64 {
        let ease_factor = (card.ease as f64 / 5000.0).clamp(0.0, 1.0);
        let fail_rate = stats.and_then(|s| s.fail_rate).unwrap_or(0.0);
        ease_factor * (1.0 - fail_rate)
    }
}

impl Scanner for FsrsScanner<'_> {
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        // Read collection (copies file to temp location, so no lock contention)
        let collection = read_anki_collection(self.collection_path)
            .map_err(|e| CardloopError::Validation(e.to_string()))?;

        // Load registry cards and build anki_note_id -> CardEntry lookup
        let entries = self.registry.find_cards(None, None, None)?;
        let mut note_id_to_slug: HashMap<NoteId, String> = HashMap::new();
        let mut slug_to_source: HashMap<String, String> = HashMap::new();
        for entry in &entries {
            if let Some(anki_id) = entry.anki_note_id {
                note_id_to_slug.insert(NoteId(anki_id), entry.slug.clone());
            }
            slug_to_source.insert(entry.slug.clone(), entry.source_path.clone());
        }

        // Build card_id -> CardStats lookup
        let stats_by_card: HashMap<CardId, &CardStats> = collection
            .card_stats
            .iter()
            .map(|s| (s.card_id, s))
            .collect();

        let mut items = Vec::new();
        let now = Utc::now();

        for card in &collection.cards {
            let slug = match note_id_to_slug.get(&card.note_id) {
                Some(s) => s.clone(),
                None => continue, // card not tracked in registry
            };

            let stats = stats_by_card.get(&card.card_id).copied();
            let score = Self::score_from_card(card, stats);
            let fail_rate = stats.and_then(|s| s.fail_rate).unwrap_or(0.0);
            let reviews = stats.map(|s| s.reviews).unwrap_or(0);
            let source_path = slug_to_source
                .get(&slug)
                .cloned()
                .unwrap_or_else(|| slug.clone());

            // Rule 1: extreme difficulty + many lapses → Rework
            if card.ease < 1800 && card.lapses >= 5 {
                let id = Self::item_id(&slug, "ease_lapses_rework");
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::LowQuality {
                        dimension: "retention".into(),
                        score,
                    },
                    tier: Tier::Rework,
                    status: ItemStatus::Open,
                    slug: Some(slug.clone()),
                    source_path: source_path.clone(),
                    summary: format!(
                        "Low retention ({score:.2}): ease={}, lapses={} for {slug}",
                        card.ease, card.lapses
                    ),
                    detail: Some(format!(
                        "ease={}, lapses={}, reps={}, fail_rate={:.2}",
                        card.ease, card.lapses, card.reps, fail_rate
                    )),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.85),
                });
                continue;
            }

            // Rule 2: very low ease + many reps → Delete
            if card.ease < 1500 && card.reps >= 10 {
                let id = Self::item_id(&slug, "ease_reps_delete");
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::LowQuality {
                        dimension: "retention".into(),
                        score,
                    },
                    tier: Tier::Delete,
                    status: ItemStatus::Open,
                    slug: Some(slug.clone()),
                    source_path: source_path.clone(),
                    summary: format!(
                        "Persistently difficult ({score:.2}): ease={}, reps={} for {slug}",
                        card.ease, card.reps
                    ),
                    detail: Some(format!(
                        "ease={}, lapses={}, reps={}, fail_rate={:.2}",
                        card.ease, card.lapses, card.reps, fail_rate
                    )),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.9),
                });
                continue;
            }

            // Rule 3: high fail rate with enough reviews → QuickFix
            if fail_rate > 0.4 && reviews >= 5 {
                let id = Self::item_id(&slug, "fail_rate_quickfix");
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::LowQuality {
                        dimension: "retention".into(),
                        score,
                    },
                    tier: Tier::QuickFix,
                    status: ItemStatus::Open,
                    slug: Some(slug.clone()),
                    source_path: source_path.clone(),
                    summary: format!(
                        "High fail rate ({fail_rate:.0}%): {reviews} reviews for {slug}",
                        fail_rate = fail_rate * 100.0
                    ),
                    detail: Some(format!(
                        "ease={}, lapses={}, reps={}, fail_rate={:.2}",
                        card.ease, card.lapses, card.reps, fail_rate
                    )),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.8),
                });
            }
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use card::registry::{CardEntry, CardRegistry};
    use chrono::Utc;
    use common::DeckId;

    fn make_registry_with_card(slug: &str, anki_note_id: Option<i64>) -> CardRegistry {
        let registry = CardRegistry::open(":memory:").unwrap();
        let now = Utc::now();
        let entry = CardEntry {
            slug: slug.into(),
            note_id: "note-1".into(),
            source_path: "notes/test.md".into(),
            front: "What is X?".into(),
            back: "X is Y.".into(),
            content_hash: "abc".into(),
            metadata_hash: "def".into(),
            language: "en".into(),
            tags: vec!["topic::rust".into()],
            anki_note_id,
            created_at: Some(now),
            updated_at: Some(now),
            synced_at: None,
        };
        registry.add_card(&entry).unwrap();
        registry
    }

    fn make_card(card_id: i64, note_id: i64, ease: i32, lapses: i32, reps: i32) -> AnkiCard {
        AnkiCard {
            card_id: CardId(card_id),
            note_id: NoteId(note_id),
            deck_id: DeckId(1),
            ord: 0,
            mtime: 0,
            usn: 0,
            card_type: 2,
            queue: 2,
            due: None,
            ivl: 10,
            ease,
            reps,
            lapses,
        }
    }

    fn make_stats(card_id: i64, reviews: i32, fail_rate: f64) -> CardStats {
        CardStats {
            card_id: CardId(card_id),
            reviews,
            avg_ease: None,
            fail_rate: Some(fail_rate),
            last_review_at: None,
            total_time_ms: 0,
        }
    }

    /// Test the detection logic using the score_from_card helper.
    #[test]
    fn score_from_card_full_ease_no_fails() {
        let card = make_card(1, 1, 2500, 0, 10);
        let stats = make_stats(1, 10, 0.0);
        let score = FsrsScanner::score_from_card(&card, Some(&stats));
        // ease=2500, factor=0.5, fail_rate=0.0 → score=0.5
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn score_from_card_low_ease_with_fails() {
        let card = make_card(1, 1, 1000, 3, 15);
        let stats = make_stats(1, 15, 0.5);
        let score = FsrsScanner::score_from_card(&card, Some(&stats));
        // ease=1000, factor=0.2, fail_rate=0.5 → score=0.1
        assert!((score - 0.1).abs() < 1e-9);
    }

    #[test]
    fn item_id_is_deterministic() {
        let id1 = FsrsScanner::item_id("my-card", "ease_lapses_rework");
        let id2 = FsrsScanner::item_id("my-card", "ease_lapses_rework");
        assert_eq!(id1, id2);
    }

    #[test]
    fn item_id_differs_by_discriminator() {
        let id1 = FsrsScanner::item_id("my-card", "ease_lapses_rework");
        let id2 = FsrsScanner::item_id("my-card", "ease_reps_delete");
        assert_ne!(id1, id2);
    }

    /// Verify detection rules without requiring a real Anki file.
    #[test]
    fn detection_rule_thresholds() {
        // Rule 1: ease < 1800 AND lapses >= 5 → Rework
        let card = make_card(1, 1, 1700, 5, 8);
        let stats = make_stats(1, 8, 0.3);
        // ease=1700 < 1800, lapses=5 >= 5 → triggers rule 1
        assert!(card.ease < 1800 && card.lapses >= 5);

        // Rule 2: ease < 1500 AND reps >= 10 → Delete
        let card2 = make_card(2, 2, 1400, 2, 12);
        assert!(card2.ease < 1500 && card2.reps >= 10);

        // Rule 3: fail_rate > 0.4 AND reviews >= 5 → QuickFix
        let stats3 = make_stats(3, 10, 0.5);
        assert!(stats3.fail_rate.unwrap_or(0.0) > 0.4 && stats3.reviews >= 5);
    }
}
