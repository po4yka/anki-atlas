use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Which loop produced this work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopKind {
    Generation,
    Audit,
}

impl LoopKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Generation => "generation",
            Self::Audit => "audit",
        }
    }
}

impl std::str::FromStr for LoopKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "generation" => Ok(Self::Generation),
            "audit" => Ok(Self::Audit),
            _ => Err(format!("unknown loop kind: {s}")),
        }
    }
}

/// Work item lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    Open,
    InProgress,
    Fixed,
    Skipped,
    WontFix,
}

impl ItemStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::InProgress => "in_progress",
            Self::Fixed => "fixed",
            Self::Skipped => "skipped",
            Self::WontFix => "wontfix",
        }
    }

    /// Whether this status counts as resolved (not in the active queue).
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Fixed | Self::Skipped | Self::WontFix)
    }
}

impl std::str::FromStr for ItemStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "open" => Ok(Self::Open),
            "in_progress" => Ok(Self::InProgress),
            "fixed" => Ok(Self::Fixed),
            "skipped" => Ok(Self::Skipped),
            "wontfix" => Ok(Self::WontFix),
            _ => Err(format!("unknown status: {s}")),
        }
    }
}

/// Issue severity tier, ordered from easiest to hardest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Tag issues, format issues (automatable).
    AutoFix = 1,
    /// Content tweaks, clarity improvements.
    QuickFix = 2,
    /// Split, rewrite, significant edits.
    Rework = 3,
    /// Dead skills, irrelevant, unfixable duplicates.
    Delete = 4,
}

impl Tier {
    pub fn as_i32(&self) -> i32 {
        *self as i32
    }

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            1 => Some(Self::AutoFix),
            2 => Some(Self::QuickFix),
            3 => Some(Self::Rework),
            4 => Some(Self::Delete),
            _ => None,
        }
    }
}

/// What kind of issue was detected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IssueKind {
    // -- Audit issues --
    LowQuality {
        dimension: String,
        score: f64,
    },
    ValidationError {
        severity: String,
        message: String,
    },
    /// Near-identical duplicate: similarity >= 0.92.
    Duplicate {
        other_slug: String,
        similarity: f64,
    },
    /// Semantic overlap: similarity in [0.82, 0.92). Softer signal than Duplicate.
    ///
    /// TODO: Wire analytics crate duplicate detector to populate these.
    SemanticOverlap {
        other_slug: String,
        similarity: f64,
    },
    SplitCandidate {
        suggested_count: u32,
    },
    StaleContent,
    DeadSkill,
    MissingTags,

    // -- Generation issues --
    UncoveredTopic {
        topic: String,
    },
    MissingLanguage {
        expected: String,
    },
}

/// A single work item in the queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub id: String,
    pub loop_kind: LoopKind,
    pub issue_kind: IssueKind,
    pub tier: Tier,
    pub status: ItemStatus,
    /// Card slug (audit loop) or None (generation loop).
    pub slug: Option<String>,
    /// Note or card source file path.
    pub source_path: String,
    /// One-line human-readable description.
    pub summary: String,
    /// Extended context for the agent.
    pub detail: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    /// Attestation: why it was fixed/skipped.
    pub attestation: Option<String>,
    /// Which scan number produced this item.
    pub scan_number: u32,
    /// Cluster this item belongs to (slug-based or kind-based).
    pub cluster_id: Option<String>,
    /// Scanner-assigned confidence in [0.0, 1.0]. None = not assessed.
    pub confidence: Option<f64>,
}

/// Immutable event for the progression log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressionEvent {
    pub timestamp: DateTime<Utc>,
    /// Action name: "scan", "resolve", "skip", "reopen".
    pub action: String,
    pub item_ids: Vec<String>,
    /// Who performed it: "agent" or "user".
    pub actor: String,
    pub note: Option<String>,
    pub scores_before: Option<ScoreSummary>,
    pub scores_after: Option<ScoreSummary>,
}

/// Dashboard score snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreSummary {
    /// Lenient score: fixed / (fixed + open). WontFix/Skipped excluded.
    pub overall: f64,
    /// Strict score: fixed / total (wontfix counts against you).
    pub strict_score: f64,
    /// Mean QualityScore.overall() across scanned cards.
    pub quality_avg: f64,
    pub open_count: usize,
    pub fixed_count: usize,
    pub skipped_count: usize,
    /// Counts per tier: [autofix, quickfix, rework, delete].
    pub by_tier: [usize; 4],
    /// (generation_open, audit_open).
    pub by_loop: (usize, usize),
    /// FSRS-based health score. Formula: (1 - D/10) * stability_growth_rate * (1 - lapse_rate).
    /// Always None until anki-reader FSRS integration is wired.
    pub health_score: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_kind_roundtrip() {
        assert_eq!(
            LoopKind::Generation.as_str().parse::<LoopKind>().unwrap(),
            LoopKind::Generation
        );
        assert_eq!(
            LoopKind::Audit.as_str().parse::<LoopKind>().unwrap(),
            LoopKind::Audit
        );
        assert!("bogus".parse::<LoopKind>().is_err());
    }

    #[test]
    fn item_status_roundtrip() {
        for status in [
            ItemStatus::Open,
            ItemStatus::InProgress,
            ItemStatus::Fixed,
            ItemStatus::Skipped,
            ItemStatus::WontFix,
        ] {
            assert_eq!(status.as_str().parse::<ItemStatus>().unwrap(), status);
        }
    }

    #[test]
    fn tier_ordering() {
        assert!(Tier::AutoFix < Tier::QuickFix);
        assert!(Tier::QuickFix < Tier::Rework);
        assert!(Tier::Rework < Tier::Delete);
    }

    #[test]
    fn tier_roundtrip() {
        for (i, tier) in [
            (1, Tier::AutoFix),
            (2, Tier::QuickFix),
            (3, Tier::Rework),
            (4, Tier::Delete),
        ] {
            assert_eq!(Tier::from_i32(i), Some(tier));
            assert_eq!(tier.as_i32(), i);
        }
        assert_eq!(Tier::from_i32(99), None);
    }

    #[test]
    fn resolved_statuses() {
        assert!(!ItemStatus::Open.is_resolved());
        assert!(!ItemStatus::InProgress.is_resolved());
        assert!(ItemStatus::Fixed.is_resolved());
        assert!(ItemStatus::Skipped.is_resolved());
        assert!(ItemStatus::WontFix.is_resolved());
    }

    #[test]
    fn issue_kind_json_roundtrip() {
        let kind = IssueKind::LowQuality {
            dimension: "clarity".into(),
            score: 0.3,
        };
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: IssueKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, parsed);
    }
}
