use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Type of relationship between concepts or topics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum EdgeType {
    /// Semantically similar content (cosine similarity 0.7-0.93).
    Similar,
    /// Must learn source before target.
    Prerequisite,
    /// Related but distinct concepts (tag co-occurrence, etc.).
    Related,
    /// Explicit cross-reference (Obsidian wikilink).
    CrossReference,
    /// Source is a more specific version of target.
    Specialization,
}

/// How the edge was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum EdgeSource {
    /// Vector similarity (embedding cosine distance).
    Embedding,
    /// Shared tags across notes.
    TagCooccurrence,
    /// Correlated failure patterns in review data.
    ReviewInference,
    /// Obsidian `[[wikilink]]`.
    Wikilink,
    /// Parent/child path relationship in taxonomy.
    Taxonomy,
    /// User-curated relationship.
    Manual,
}

/// A directed edge between two notes (concept-level).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptEdge {
    pub source_note_id: i64,
    pub target_note_id: i64,
    pub edge_type: EdgeType,
    pub edge_source: EdgeSource,
    /// Strength of the relationship (0.0-1.0).
    pub weight: f32,
}

/// A directed edge between two topics (topic-level).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicEdge {
    pub source_topic_id: i32,
    pub target_topic_id: i32,
    pub edge_type: EdgeType,
    pub edge_source: EdgeSource,
    /// Strength of the relationship (0.0-1.0).
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn edge_type_roundtrip() {
        assert_eq!(EdgeType::from_str("similar").unwrap(), EdgeType::Similar);
        assert_eq!(
            EdgeType::from_str("prerequisite").unwrap(),
            EdgeType::Prerequisite
        );
        assert_eq!(EdgeType::Similar.to_string(), "similar");
    }

    #[test]
    fn edge_source_roundtrip() {
        assert_eq!(
            EdgeSource::from_str("embedding").unwrap(),
            EdgeSource::Embedding
        );
        assert_eq!(
            EdgeSource::from_str("tag_cooccurrence").unwrap(),
            EdgeSource::TagCooccurrence
        );
        assert_eq!(EdgeSource::Embedding.to_string(), "embedding");
    }

    #[test]
    fn concept_edge_json_roundtrip() {
        let edge = ConceptEdge {
            source_note_id: 1,
            target_note_id: 2,
            edge_type: EdgeType::Similar,
            edge_source: EdgeSource::Embedding,
            weight: 0.85,
        };
        let json = serde_json::to_string(&edge).unwrap();
        let restored: ConceptEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(edge, restored);
    }
}
