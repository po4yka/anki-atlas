use anki_atlas_mcp::formatters::{
    format_duplicates, format_job_accepted, format_job_status, format_search, format_topics,
    format_workflow,
};
use anki_atlas_mcp::tools::{
    DuplicatesToolResult, JobAcceptedToolResult, JobStatusToolResult, SearchResultView,
    SearchToolResult, TopicsToolResult, WorkflowToolResult,
};

#[test]
fn search_formatter_includes_query_and_results() {
    let formatted = format_search(&SearchToolResult {
        query: "ownership".to_string(),
        total_results: 1,
        lexical_mode: "fts".to_string(),
        lexical_fallback_used: false,
        rerank_applied: true,
        query_suggestions: vec!["ownership and borrowing".to_string()],
        autocomplete_suggestions: vec!["ownership".to_string()],
        results: vec![SearchResultView {
            note_id: 1,
            rrf_score: 0.95,
            semantic_score: Some(0.9),
            fts_score: Some(0.8),
            rerank_score: Some(0.97),
            headline: Some("Ownership".to_string()),
            sources: vec!["semantic".to_string(), "fts".to_string()],
        }],
    });
    assert!(formatted.contains("ownership"));
    assert!(formatted.contains("note `1`"));
}

#[test]
fn topics_formatter_includes_root_path() {
    let formatted = format_topics(&TopicsToolResult {
        root_path: Some("rust".to_string()),
        topic_count: 3,
        topics: serde_json::json!([]),
    });
    assert!(formatted.contains("rust"));
    assert!(formatted.contains("3"));
}

#[test]
fn duplicates_formatter_mentions_threshold() {
    let formatted = format_duplicates(&DuplicatesToolResult {
        threshold: 0.92,
        max_clusters: 10,
        clusters: serde_json::json!([]),
        stats: serde_json::json!({"clusters_found":0}),
    });
    assert!(formatted.contains("0.92"));
}

#[test]
fn job_formatters_include_ids() {
    let accepted = format_job_accepted(&JobAcceptedToolResult {
        job_id: "job-1".to_string(),
        job_type: "sync".to_string(),
        status: "queued".to_string(),
        poll_hint: "poll".to_string(),
        cancel_hint: "cancel".to_string(),
    });
    let status = format_job_status(&JobStatusToolResult {
        job_id: "job-1".to_string(),
        job_type: "sync".to_string(),
        status: "running".to_string(),
        progress: 0.5,
        message: Some("halfway".to_string()),
        result: None,
        error: None,
    });
    assert!(accepted.contains("job-1"));
    assert!(status.contains("running"));
}

#[test]
fn workflow_formatter_mentions_path() {
    let formatted = format_workflow(&WorkflowToolResult {
        path: "/tmp/file.md".to_string(),
        summary: "estimated cards: 2".to_string(),
        data: serde_json::json!({}),
    });
    assert!(formatted.contains("/tmp/file.md"));
}
