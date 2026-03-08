use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::AnalyticsError;

/// A topic node in the taxonomy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Slash-delimited path (e.g. "programming/python/async").
    pub path: String,
    /// Human-readable label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
    /// Database ID (populated after DB insert).
    pub topic_id: Option<i64>,
    /// Child topics (populated during tree construction).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Topic>,
}

impl Topic {
    /// Parent path (everything before last '/'), or None for root topics.
    pub fn parent_path(&self) -> Option<&str> {
        todo!()
    }

    /// Depth = number of '/' in path.
    pub fn depth(&self) -> usize {
        todo!()
    }

    /// Short name = last segment of path.
    pub fn name(&self) -> &str {
        todo!()
    }
}

/// Complete taxonomy with lookup index.
#[derive(Debug, Clone, Default)]
pub struct Taxonomy {
    /// path -> Topic lookup.
    pub topics: HashMap<String, Topic>,
    /// Root topics (depth 0).
    pub roots: Vec<Topic>,
}

impl Taxonomy {
    /// Get topic by path.
    pub fn get(&self, path: &str) -> Option<&Topic> {
        todo!()
    }

    /// All topics in depth-first order.
    pub fn all_topics(&self) -> Vec<&Topic> {
        todo!()
    }

    /// All topics under a given path (inclusive).
    pub fn subtree(&self, path: &str) -> Vec<&Topic> {
        todo!()
    }
}

/// Load taxonomy from a YAML file.
pub fn load_taxonomy_from_yaml(path: &std::path::Path) -> Result<Taxonomy, AnalyticsError> {
    todo!()
}

/// Sync taxonomy to database (upsert by path). Returns path -> topic_id map.
pub async fn sync_taxonomy_to_db(
    _pool: &sqlx::PgPool,
    _taxonomy: &Taxonomy,
) -> Result<HashMap<String, i64>, AnalyticsError> {
    todo!()
}

/// Load taxonomy from database, reconstructing tree structure.
pub async fn load_taxonomy_from_db(
    _pool: &sqlx::PgPool,
) -> Result<Taxonomy, AnalyticsError> {
    todo!()
}

/// Get a single topic by path from database.
pub async fn get_topic_by_path(
    _pool: &sqlx::PgPool,
    _path: &str,
) -> Result<Option<Topic>, AnalyticsError> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topic(path: &str, label: &str) -> Topic {
        Topic {
            path: path.to_string(),
            label: label.to_string(),
            description: None,
            topic_id: None,
            children: vec![],
        }
    }

    // --- Topic methods ---

    #[test]
    fn topic_parent_path_returns_none_for_root() {
        let topic = make_topic("programming", "Programming");
        assert_eq!(topic.parent_path(), None);
    }

    #[test]
    fn topic_parent_path_returns_parent_for_nested() {
        let topic = make_topic("programming/python", "Python");
        assert_eq!(topic.parent_path(), Some("programming"));
    }

    #[test]
    fn topic_parent_path_returns_parent_for_deeply_nested() {
        let topic = make_topic("programming/python/async", "Async");
        assert_eq!(topic.parent_path(), Some("programming/python"));
    }

    #[test]
    fn topic_depth_root_is_zero() {
        let topic = make_topic("programming", "Programming");
        assert_eq!(topic.depth(), 0);
    }

    #[test]
    fn topic_depth_one_level() {
        let topic = make_topic("programming/python", "Python");
        assert_eq!(topic.depth(), 1);
    }

    #[test]
    fn topic_depth_two_levels() {
        let topic = make_topic("a/b/c", "C");
        assert_eq!(topic.depth(), 2);
    }

    #[test]
    fn topic_name_root() {
        let topic = make_topic("programming", "Programming");
        assert_eq!(topic.name(), "programming");
    }

    #[test]
    fn topic_name_nested() {
        let topic = make_topic("programming/python", "Python");
        assert_eq!(topic.name(), "python");
    }

    // --- Taxonomy methods ---

    fn make_taxonomy() -> Taxonomy {
        let mut taxonomy = Taxonomy::default();
        let root = make_topic("a", "A");
        let child = make_topic("a/b", "B");
        let grandchild = make_topic("a/b/c", "C");
        let other_root = make_topic("x", "X");
        // Separate tree to test subtree doesn't match prefix "ab"
        let ab_trap = make_topic("ab", "AB");

        taxonomy.topics.insert("a".to_string(), root.clone());
        taxonomy.topics.insert("a/b".to_string(), child.clone());
        taxonomy.topics.insert("a/b/c".to_string(), grandchild.clone());
        taxonomy.topics.insert("x".to_string(), other_root.clone());
        taxonomy.topics.insert("ab".to_string(), ab_trap.clone());

        // Build tree: roots have children wired
        let mut root_with_children = root;
        let mut child_with_gc = child;
        child_with_gc.children = vec![grandchild];
        root_with_children.children = vec![child_with_gc];

        taxonomy.roots = vec![root_with_children, other_root, ab_trap];
        taxonomy
    }

    #[test]
    fn taxonomy_get_existing() {
        let taxonomy = make_taxonomy();
        let topic = taxonomy.get("a/b");
        assert!(topic.is_some());
        assert_eq!(topic.unwrap().label, "B");
    }

    #[test]
    fn taxonomy_get_missing() {
        let taxonomy = make_taxonomy();
        assert!(taxonomy.get("nonexistent").is_none());
    }

    #[test]
    fn taxonomy_all_topics_depth_first() {
        let taxonomy = make_taxonomy();
        let all = taxonomy.all_topics();
        // Should include all topics from roots in depth-first order
        assert!(all.len() >= 5);
        // First root "a" should come before its children
        let paths: Vec<&str> = all.iter().map(|t| t.path.as_str()).collect();
        let a_pos = paths.iter().position(|p| *p == "a").unwrap();
        let ab_pos = paths.iter().position(|p| *p == "a/b").unwrap();
        let abc_pos = paths.iter().position(|p| *p == "a/b/c").unwrap();
        assert!(a_pos < ab_pos);
        assert!(ab_pos < abc_pos);
    }

    #[test]
    fn taxonomy_subtree_includes_self_and_descendants() {
        let taxonomy = make_taxonomy();
        let subtree = taxonomy.subtree("a");
        let paths: Vec<&str> = subtree.iter().map(|t| t.path.as_str()).collect();
        assert!(paths.contains(&"a"));
        assert!(paths.contains(&"a/b"));
        assert!(paths.contains(&"a/b/c"));
    }

    #[test]
    fn taxonomy_subtree_excludes_prefix_match() {
        let taxonomy = make_taxonomy();
        let subtree = taxonomy.subtree("a");
        let paths: Vec<&str> = subtree.iter().map(|t| t.path.as_str()).collect();
        // "ab" should NOT be in subtree of "a"
        assert!(!paths.contains(&"ab"));
    }

    #[test]
    fn taxonomy_subtree_excludes_unrelated() {
        let taxonomy = make_taxonomy();
        let subtree = taxonomy.subtree("a");
        let paths: Vec<&str> = subtree.iter().map(|t| t.path.as_str()).collect();
        assert!(!paths.contains(&"x"));
    }

    // --- YAML loading ---

    #[test]
    fn load_taxonomy_from_yaml_valid() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("taxonomy.yaml");
        std::fs::write(
            &yaml_path,
            r#"
topics:
  - path: programming
    label: Programming
    description: General programming concepts
    children:
      - path: programming/python
        label: Python
        children:
          - path: programming/python/async
            label: Async Programming
  - path: math
    label: Mathematics
"#,
        )
        .unwrap();

        let taxonomy = load_taxonomy_from_yaml(&yaml_path).unwrap();
        assert_eq!(taxonomy.topics.len(), 4);
        assert_eq!(taxonomy.roots.len(), 2);
        assert!(taxonomy.topics.contains_key("programming/python/async"));
    }

    #[test]
    fn load_taxonomy_from_yaml_missing_file_returns_empty() {
        let path = std::path::Path::new("/nonexistent/taxonomy.yaml");
        let taxonomy = load_taxonomy_from_yaml(path).unwrap();
        assert!(taxonomy.topics.is_empty());
        assert!(taxonomy.roots.is_empty());
    }

    #[test]
    fn load_taxonomy_from_yaml_empty_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("empty.yaml");
        std::fs::write(&yaml_path, "").unwrap();

        let taxonomy = load_taxonomy_from_yaml(&yaml_path).unwrap();
        assert!(taxonomy.topics.is_empty());
    }

    #[test]
    fn load_taxonomy_from_yaml_no_topics_key_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("bad.yaml");
        std::fs::write(&yaml_path, "something_else: true\n").unwrap();

        let taxonomy = load_taxonomy_from_yaml(&yaml_path).unwrap();
        assert!(taxonomy.topics.is_empty());
    }

    #[test]
    fn load_taxonomy_from_yaml_wires_children() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("taxonomy.yaml");
        std::fs::write(
            &yaml_path,
            r#"
topics:
  - path: a
    label: A
    children:
      - path: a/b
        label: B
"#,
        )
        .unwrap();

        let taxonomy = load_taxonomy_from_yaml(&yaml_path).unwrap();
        assert_eq!(taxonomy.roots.len(), 1);
        assert_eq!(taxonomy.roots[0].children.len(), 1);
        assert_eq!(taxonomy.roots[0].children[0].path, "a/b");
    }

    #[test]
    fn load_taxonomy_from_yaml_uses_last_segment_as_default_label() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("taxonomy.yaml");
        std::fs::write(
            &yaml_path,
            r#"
topics:
  - path: programming/python
"#,
        )
        .unwrap();

        let taxonomy = load_taxonomy_from_yaml(&yaml_path).unwrap();
        let topic = taxonomy.topics.get("programming/python").unwrap();
        assert_eq!(topic.label, "python");
    }
}
