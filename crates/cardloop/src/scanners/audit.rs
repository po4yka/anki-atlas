use card::registry::CardRegistry;
use chrono::Utc;
use validation::pipeline::{Severity, ValidationPipeline};
use validation::quality::assess_quality_with_tags;

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::Scanner;

/// Minimum quality dimension score before flagging.
const QUALITY_THRESHOLD: f64 = 0.5;

/// Scans existing cards in a `CardRegistry` for quality issues.
pub struct AuditScanner<'a> {
    registry: &'a CardRegistry,
    pipeline: &'a ValidationPipeline,
}

impl<'a> AuditScanner<'a> {
    pub fn new(registry: &'a CardRegistry, pipeline: &'a ValidationPipeline) -> Self {
        Self { registry, pipeline }
    }

    /// Generate a deterministic ID for deduplication: same slug + issue kind = same ID.
    fn item_id(slug: &str, discriminator: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(slug.as_bytes());
        hasher.update(b":");
        hasher.update(discriminator.as_bytes());
        let hash = hasher.finalize();
        // Use first 16 hex chars as a stable ID
        hex_prefix(&hash)
    }
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

impl Scanner for AuditScanner<'_> {
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        let cards = self.registry.find_cards(None, None, None)?;
        let mut items = Vec::new();
        let now = Utc::now();

        for card in &cards {
            // 1. Run validation pipeline
            let result = self.pipeline.run(&card.front, &card.back, &card.tags);
            for issue in &result.issues {
                if issue.severity == Severity::Info {
                    continue;
                }
                let tier = match issue.severity {
                    Severity::Error => Tier::QuickFix,
                    Severity::Warning => Tier::AutoFix,
                    Severity::Info => continue,
                };
                let id =
                    AuditScanner::item_id(&card.slug, &format!("validation:{}", issue.message));
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::ValidationError {
                        severity: issue.severity.to_string(),
                        message: issue.message.clone(),
                    },
                    tier,
                    status: ItemStatus::Open,
                    slug: Some(card.slug.clone()),
                    source_path: card.source_path.clone(),
                    summary: format!("[{}] {}: {}", issue.severity, card.slug, issue.message),
                    detail: Some(format!("location: {}", issue.location)),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(1.0),
                });
            }

            // 2. Quality scoring
            let quality = assess_quality_with_tags(&card.front, &card.back, &card.tags);
            for (dimension, score) in [
                ("clarity", quality.clarity),
                ("atomicity", quality.atomicity),
                ("testability", quality.testability),
                ("memorability", quality.memorability),
                ("accuracy", quality.accuracy),
                ("relevance", quality.relevance),
            ] {
                if score < QUALITY_THRESHOLD {
                    let id = AuditScanner::item_id(&card.slug, &format!("quality:{dimension}"));
                    items.push(WorkItem {
                        id,
                        loop_kind: LoopKind::Audit,
                        issue_kind: IssueKind::LowQuality {
                            dimension: dimension.to_string(),
                            score,
                        },
                        tier: Tier::QuickFix,
                        status: ItemStatus::Open,
                        slug: Some(card.slug.clone()),
                        source_path: card.source_path.clone(),
                        summary: format!(
                            "Low {dimension} ({score:.2}) for {slug}",
                            slug = card.slug
                        ),
                        detail: None,
                        first_seen: now,
                        resolved_at: None,
                        attestation: None,
                        scan_number,
                        cluster_id: None,
                        confidence: Some(0.8),
                    });
                }
            }

            // 3. Dead skill detection
            if card.tags.iter().any(|t| t == "skill::dead") {
                let id = AuditScanner::item_id(&card.slug, "dead_skill");
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::DeadSkill,
                    tier: Tier::Delete,
                    status: ItemStatus::Open,
                    slug: Some(card.slug.clone()),
                    source_path: card.source_path.clone(),
                    summary: format!("Dead skill: {}", card.slug),
                    detail: None,
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.9),
                });
            }

            // 4. Missing tags
            let has_topic = card.tags.iter().any(|t| t.starts_with("topic::"));
            let has_skill = card.tags.iter().any(|t| t.starts_with("skill::"));
            if !has_topic && !has_skill {
                let id = AuditScanner::item_id(&card.slug, "missing_tags");
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::MissingTags,
                    tier: Tier::AutoFix,
                    status: ItemStatus::Open,
                    slug: Some(card.slug.clone()),
                    source_path: card.source_path.clone(),
                    summary: format!("Missing topic/skill tags: {}", card.slug),
                    detail: None,
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(1.0),
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
    use validation::{
        ContentValidator, FormatValidator, HtmlValidator, RelevanceValidator, TagValidator,
    };

    fn test_registry() -> CardRegistry {
        CardRegistry::open(":memory:").unwrap()
    }

    fn test_pipeline() -> ValidationPipeline {
        ValidationPipeline::new(vec![
            Box::new(ContentValidator::default()),
            Box::new(FormatValidator),
            Box::new(HtmlValidator),
            Box::new(TagValidator::default()),
            Box::new(RelevanceValidator::new()),
        ])
    }

    fn make_card(slug: &str, front: &str, back: &str, tags: Vec<String>) -> CardEntry {
        let now = Utc::now();
        CardEntry {
            slug: slug.into(),
            note_id: "note-1".into(),
            source_path: "notes/test.md".into(),
            front: front.into(),
            back: back.into(),
            content_hash: "abc123".into(),
            metadata_hash: "def456".into(),
            language: "en".into(),
            tags,
            anki_note_id: None,
            created_at: Some(now),
            updated_at: Some(now),
            synced_at: None,
        }
    }

    #[test]
    fn detects_missing_tags() {
        let registry = test_registry();
        let pipeline = test_pipeline();

        let card = make_card(
            "test-no-tags",
            "What is ownership in Rust?",
            "Ownership is a set of rules governing memory management.",
            vec![], // no topic:: or skill:: tags
        );
        registry.add_card(&card).unwrap();

        let scanner = AuditScanner::new(&registry, &pipeline);
        let items = scanner.scan(1).unwrap();

        let missing_tag_items: Vec<_> = items
            .iter()
            .filter(|i| matches!(i.issue_kind, IssueKind::MissingTags))
            .collect();
        assert!(!missing_tag_items.is_empty(), "should detect missing tags");
        assert_eq!(missing_tag_items[0].tier, Tier::AutoFix);
    }

    #[test]
    fn detects_dead_skill() {
        let registry = test_registry();
        let pipeline = test_pipeline();

        let card = make_card(
            "test-dead",
            "What is jQuery.ajax()?",
            "jQuery.ajax() performs asynchronous HTTP requests.",
            vec!["skill::dead".into(), "topic::javascript".into()],
        );
        registry.add_card(&card).unwrap();

        let scanner = AuditScanner::new(&registry, &pipeline);
        let items = scanner.scan(1).unwrap();

        let dead_items: Vec<_> = items
            .iter()
            .filter(|i| matches!(i.issue_kind, IssueKind::DeadSkill))
            .collect();
        assert!(!dead_items.is_empty(), "should detect dead skill");
        assert_eq!(dead_items[0].tier, Tier::Delete);
    }

    #[test]
    fn skips_healthy_card() {
        let registry = test_registry();
        let pipeline = test_pipeline();

        let card = make_card(
            "test-healthy",
            "What is the borrow checker in Rust?",
            "The borrow checker enforces ownership rules at compile time, preventing data races and dangling references.",
            vec!["topic::rust".into(), "skill::alive".into()],
        );
        registry.add_card(&card).unwrap();

        let scanner = AuditScanner::new(&registry, &pipeline);
        let items = scanner.scan(1).unwrap();

        // A healthy card with good tags should produce minimal issues
        let error_items: Vec<_> = items
            .iter()
            .filter(|i| matches!(i.issue_kind, IssueKind::MissingTags | IssueKind::DeadSkill))
            .collect();
        assert!(
            error_items.is_empty(),
            "healthy card should not flag tag/skill issues"
        );
    }

    #[test]
    fn deterministic_ids() {
        // Same slug + discriminator should produce the same ID
        let id1 = AuditScanner::item_id("my-card", "quality:clarity");
        let id2 = AuditScanner::item_id("my-card", "quality:clarity");
        assert_eq!(id1, id2);

        // Different discriminator = different ID
        let id3 = AuditScanner::item_id("my-card", "quality:atomicity");
        assert_ne!(id1, id3);
    }

    #[test]
    fn empty_registry_produces_no_items() {
        let registry = test_registry();
        let pipeline = test_pipeline();
        let scanner = AuditScanner::new(&registry, &pipeline);
        let items = scanner.scan(1).unwrap();
        assert!(items.is_empty());
    }
}
