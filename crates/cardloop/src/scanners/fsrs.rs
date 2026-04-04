use std::collections::HashMap;
use std::path::Path;

use anki_reader::models::{AnkiCard, CardStats};
use anki_reader::read_anki_collection;
use card::registry::CardRegistry;
use chrono::Utc;
use common::{CardId, NoteId};
use fsrs::{FSRS, MemoryState};

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::Scanner;

/// FSRS decay constant used for retrievability calculation.
const FSRS_DECAY: f32 = -0.5;

/// Scans an Anki collection via FSRS memory-state analysis and cross-references the card registry.
///
/// Uses the `fsrs` crate to compute stability and difficulty from SM-2 parameters,
/// then applies retention-based detection rules to generate work items.
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

    /// Compute a quality score from FSRS memory state.
    ///
    /// Returns a 0.0-1.0 score combining normalized stability and inverted difficulty.
    /// Higher = healthier card.
    fn score_from_memory_state(state: &MemoryState) -> f64 {
        // Stability: days until recall drops to 90%. Typical range 1-365+.
        // Normalize: 30 days stability = 0.5, 100+ days = ~1.0
        let stability_score = (state.stability as f64 / 100.0).clamp(0.0, 1.0);

        // Difficulty: 1.0 (easy) to 10.0 (hard). Invert so easy = high score.
        let difficulty_score = (1.0 - (state.difficulty as f64 - 1.0) / 9.0).clamp(0.0, 1.0);

        // Weighted: stability matters more (70%) than difficulty (30%)
        stability_score * 0.7 + difficulty_score * 0.3
    }

    /// Fallback score using legacy ease/fail_rate when FSRS computation fails.
    fn score_from_card(card: &AnkiCard, stats: Option<&CardStats>) -> f64 {
        let ease_factor = (card.ease as f64 / 5000.0).clamp(0.0, 1.0);
        let fail_rate = stats.and_then(|s| s.fail_rate).unwrap_or(0.0);
        ease_factor * (1.0 - fail_rate)
    }

    /// Compute FSRS memory state from an AnkiCard's SM-2 parameters.
    ///
    /// Uses `memory_state_from_sm2` to bootstrap FSRS state from ease factor and interval.
    fn compute_memory_state(fsrs_engine: &FSRS, card: &AnkiCard) -> Option<MemoryState> {
        if card.ivl <= 0 || card.ease <= 0 {
            return None;
        }
        // Anki stores ease as integer (2500 = 2.5 ease factor)
        let ease_factor = card.ease as f32 / 1000.0;
        let interval = card.ivl as f32;
        let target_retention = 0.9; // SM-2 default

        fsrs_engine
            .memory_state_from_sm2(ease_factor, interval, target_retention)
            .ok()
    }

    /// Estimate days since last review from card stats.
    fn days_since_review(stats: Option<&CardStats>) -> u32 {
        stats
            .and_then(|s| s.last_review_at)
            .map(|last| {
                let elapsed = Utc::now() - last;
                elapsed.num_days().max(0) as u32
            })
            .unwrap_or(0)
    }
}

impl Scanner for FsrsScanner<'_> {
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        let collection = read_anki_collection(self.collection_path)
            .map_err(|e| CardloopError::Validation(e.to_string()))?;

        let fsrs_engine = FSRS::new(Some(&[]))
            .map_err(|e| CardloopError::Validation(format!("FSRS init failed: {e}")))?;

        // Build registry lookups
        let entries = self.registry.find_cards(None, None, None)?;
        let mut note_id_to_slug: HashMap<NoteId, String> = HashMap::new();
        let mut slug_to_source: HashMap<String, String> = HashMap::new();
        for entry in &entries {
            if let Some(anki_id) = entry.anki_note_id {
                note_id_to_slug.insert(NoteId(anki_id), entry.slug.clone());
            }
            slug_to_source.insert(entry.slug.clone(), entry.source_path.clone());
        }

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
                None => continue,
            };

            let stats = stats_by_card.get(&card.card_id).copied();
            let source_path = slug_to_source
                .get(&slug)
                .cloned()
                .unwrap_or_else(|| slug.clone());

            // Compute FSRS memory state, fall back to legacy scoring
            let (score, stability, difficulty, retrievability) =
                match Self::compute_memory_state(&fsrs_engine, card) {
                    Some(state) => {
                        let days_elapsed = Self::days_since_review(stats);
                        let r = fsrs_engine.current_retrievability(state, days_elapsed, FSRS_DECAY);
                        (
                            Self::score_from_memory_state(&state),
                            Some(state.stability),
                            Some(state.difficulty),
                            Some(r),
                        )
                    }
                    None => (Self::score_from_card(card, stats), None, None, None),
                };

            let detail_str = format!(
                "stability={:.1}d, difficulty={:.1}, retrievability={:.0}%, ease={}, lapses={}, reps={}",
                stability.unwrap_or(0.0),
                difficulty.unwrap_or(0.0),
                retrievability.unwrap_or(0.0) * 100.0,
                card.ease,
                card.lapses,
                card.reps
            );

            // Rule 1: Low stability + high difficulty → Rework
            // FSRS: stability < 5 days AND difficulty > 7.0
            // Fallback: ease < 1800 AND lapses >= 5
            let needs_rework = match (stability, difficulty) {
                (Some(s), Some(d)) => s < 5.0 && d > 7.0,
                _ => card.ease < 1800 && card.lapses >= 5,
            };

            if needs_rework {
                let id = Self::item_id(&slug, "fsrs_low_stability_rework");
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
                        "Low retention ({score:.2}): stability={:.1}d, difficulty={:.1} for {slug}",
                        stability.unwrap_or(0.0),
                        difficulty.unwrap_or(0.0)
                    ),
                    detail: Some(detail_str.clone()),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.85),
                });
                continue;
            }

            // Rule 2: Very low stability + many reps → Delete
            // FSRS: stability < 2 days AND difficulty > 8.5 AND reps >= 10
            // Fallback: ease < 1500 AND reps >= 10
            let needs_delete = match (stability, difficulty) {
                (Some(s), Some(d)) => s < 2.0 && d > 8.5 && card.reps >= 10,
                _ => card.ease < 1500 && card.reps >= 10,
            };

            if needs_delete {
                let id = Self::item_id(&slug, "fsrs_persistently_difficult_delete");
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
                        "Persistently difficult ({score:.2}): stability={:.1}d, reps={} for {slug}",
                        stability.unwrap_or(0.0),
                        card.reps
                    ),
                    detail: Some(detail_str.clone()),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.9),
                });
                continue;
            }

            // Rule 3: Low predicted retrievability → QuickFix
            // FSRS: retrievability < 0.5 (predicted recall < 50%)
            // Fallback: fail_rate > 0.4 AND reviews >= 5
            let fail_rate = stats.and_then(|s| s.fail_rate).unwrap_or(0.0);
            let reviews = stats.map(|s| s.reviews).unwrap_or(0);

            let needs_quickfix = match retrievability {
                Some(r) => r < 0.5 && reviews >= 5,
                None => fail_rate > 0.4 && reviews >= 5,
            };

            if needs_quickfix {
                let id = Self::item_id(&slug, "fsrs_low_retrievability_quickfix");
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
                        "Low retrievability ({:.0}%): {reviews} reviews for {slug}",
                        retrievability.unwrap_or(fail_rate as f32) * 100.0
                    ),
                    detail: Some(detail_str),
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
    use common::DeckId;

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

    #[test]
    fn fsrs_memory_state_from_sm2() {
        let fsrs_engine = FSRS::new(Some(&[])).unwrap();
        let card = make_card(1, 1, 2500, 0, 10);
        let state = FsrsScanner::compute_memory_state(&fsrs_engine, &card);
        assert!(state.is_some());
        let state = state.unwrap();
        assert!(state.stability > 0.0);
        assert!(state.difficulty >= 1.0 && state.difficulty <= 10.0);
    }

    #[test]
    fn fsrs_memory_state_skips_new_cards() {
        let fsrs_engine = FSRS::new(Some(&[])).unwrap();
        // ivl=0 means card hasn't been reviewed
        let card = make_card(1, 1, 0, 0, 0);
        let state = FsrsScanner::compute_memory_state(&fsrs_engine, &card);
        assert!(state.is_none());
    }

    #[test]
    fn score_from_memory_state_easy_stable() {
        let state = MemoryState {
            stability: 100.0,
            difficulty: 2.0,
        };
        let score = FsrsScanner::score_from_memory_state(&state);
        // stability=100 → 1.0*0.7=0.7, difficulty=2 → (1-(1/9))*0.3≈0.267
        assert!(
            score > 0.9,
            "Expected high score for easy stable card, got {score}"
        );
    }

    #[test]
    fn score_from_memory_state_hard_unstable() {
        let state = MemoryState {
            stability: 1.0,
            difficulty: 9.0,
        };
        let score = FsrsScanner::score_from_memory_state(&state);
        // stability=1 → 0.01*0.7=0.007, difficulty=9 → (1-8/9)*0.3≈0.033
        assert!(
            score < 0.1,
            "Expected low score for hard unstable card, got {score}"
        );
    }

    #[test]
    fn legacy_fallback_score() {
        let card = make_card(1, 1, 2500, 0, 10);
        let stats = make_stats(1, 10, 0.0);
        let score = FsrsScanner::score_from_card(&card, Some(&stats));
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn item_id_is_deterministic() {
        let id1 = FsrsScanner::item_id("my-card", "fsrs_low_stability_rework");
        let id2 = FsrsScanner::item_id("my-card", "fsrs_low_stability_rework");
        assert_eq!(id1, id2);
    }

    #[test]
    fn item_id_differs_by_discriminator() {
        let id1 = FsrsScanner::item_id("my-card", "fsrs_low_stability_rework");
        let id2 = FsrsScanner::item_id("my-card", "fsrs_persistently_difficult_delete");
        assert_ne!(id1, id2);
    }
}
