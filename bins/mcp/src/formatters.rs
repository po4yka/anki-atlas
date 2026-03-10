use crate::tools::{
    DuplicatesToolResult, JobAcceptedToolResult, JobStatusToolResult, SearchToolResult,
    TopicCoverageToolResult, TopicGapsToolResult, TopicWeakNotesToolResult, TopicsToolResult,
    WorkflowToolResult,
};

pub fn format_search(result: &SearchToolResult) -> String {
    let mut out = format!(
        "## Search\n\nQuery: `{}`\nResults: {}\nLexical mode: `{}`\n",
        result.query, result.total_results, result.lexical_mode
    );
    for item in &result.results {
        out.push_str(&format!(
            "- note `{}` score `{:.4}` sources `{}` {}\n",
            item.note_id,
            item.rrf_score,
            item.sources.join(","),
            item.headline.as_deref().unwrap_or("")
        ));
    }
    if !result.query_suggestions.is_empty() {
        out.push_str(&format!(
            "\nSuggestions: {}\n",
            result.query_suggestions.join(", ")
        ));
    }
    out
}

pub fn format_topics(result: &TopicsToolResult) -> String {
    format!(
        "## Topics\n\nRoot path: `{}`\nTopic entries: {}\n",
        result.root_path.as_deref().unwrap_or("/"),
        result.topic_count
    )
}

pub fn format_coverage(result: &TopicCoverageToolResult) -> String {
    if result.found {
        format!(
            "## Topic Coverage\n\nTopic: `{}`\nFound coverage data.\n",
            result.topic_path
        )
    } else {
        format!(
            "## Topic Coverage\n\nTopic: `{}`\nTopic not found.\n",
            result.topic_path
        )
    }
}

pub fn format_gaps(result: &TopicGapsToolResult) -> String {
    format!(
        "## Topic Gaps\n\nTopic: `{}`\nThreshold: `{}`\n",
        result.topic_path, result.min_coverage
    )
}

pub fn format_weak_notes(result: &TopicWeakNotesToolResult) -> String {
    format!(
        "## Weak Notes\n\nTopic: `{}`\nMax results: `{}`\n",
        result.topic_path, result.max_results
    )
}

pub fn format_duplicates(result: &DuplicatesToolResult) -> String {
    format!(
        "## Duplicates\n\nThreshold: `{}`\nMax clusters: `{}`\n",
        result.threshold, result.max_clusters
    )
}

pub fn format_job_accepted(result: &JobAcceptedToolResult) -> String {
    format!(
        "## Job Accepted\n\nJob id: `{}`\nType: `{}`\nStatus: `{}`\nPoll: `{}`\nCancel: `{}`\n",
        result.job_id, result.job_type, result.status, result.poll_hint, result.cancel_hint
    )
}

pub fn format_job_status(result: &JobStatusToolResult) -> String {
    format!(
        "## Job Status\n\nJob id: `{}`\nType: `{}`\nStatus: `{}`\nProgress: `{:.2}`\n",
        result.job_id, result.job_type, result.status, result.progress
    )
}

pub fn format_workflow(result: &WorkflowToolResult) -> String {
    format!(
        "## Workflow Result\n\nPath: `{}`\n{}\n",
        result.path, result.summary
    )
}
