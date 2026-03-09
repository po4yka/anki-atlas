use std::fmt::Write;

/// A single tag audit entry: (tag_name, issues, suggested_fix, close_matches).
pub type TagAuditEntry = (String, Vec<String>, Option<String>, Vec<String>);

/// Truncate text with ellipsis at max_len. Replaces newlines with spaces and trims.
pub fn truncate(text: &str, max_len: usize) -> String {
    let cleaned = text.replace('\n', " ");
    let trimmed = cleaned.trim();

    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        let end = max_len.saturating_sub(3);
        format!("{}...", &trimmed[..end])
    }
}

/// Format note parse result for generation preview.
pub fn format_generate_result(
    title: Option<&str>,
    sections: &[(String, String)],
    body_length: usize,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Generation Preview\n");
    let title_display = title.unwrap_or("*(not detected)*");
    let _ = writeln!(out, "**Title**: {title_display}");
    let _ = writeln!(out, "**Body length**: {body_length} chars");
    let _ = writeln!(out, "**Sections**: {}", sections.len());

    let estimated = if sections.is_empty() {
        1
    } else {
        sections.len()
    };
    let _ = writeln!(out, "**Estimated cards**: ~{estimated}");

    for (name, _content) in sections {
        let _ = writeln!(out, "- {name}");
    }

    out
}

/// Format obsidian vault scan result.
pub fn format_obsidian_sync_result(
    notes_found: usize,
    parsed_notes: &[(String, Option<String>, usize)],
    vault_path: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Obsidian Vault Scan\n");
    let _ = writeln!(out, "**Vault**: {vault_path}");
    let _ = writeln!(out, "**Notes found**: {notes_found}");

    let display_limit = 20;
    for (filename, title, _card_count) in parsed_notes.iter().take(display_limit) {
        let title_display = title.as_deref().unwrap_or("*(untitled)*");
        let _ = writeln!(out, "- **{filename}**: {title_display}");
    }

    if parsed_notes.len() > display_limit {
        let remaining = parsed_notes.len() - display_limit;
        let _ = writeln!(out, "\n...and {remaining} more notes");
    }

    out
}

/// Format tag audit result.
pub fn format_tag_audit_result(results: &[TagAuditEntry]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Tag Audit Results\n");

    let total = results.len();
    let with_issues = results.iter().filter(|(_, issues, _, _)| !issues.is_empty()).count();
    let valid = total - with_issues;

    let _ = writeln!(out, "**Total tags**: {total}");
    let _ = writeln!(out, "**Valid**: {valid}");
    let _ = writeln!(out, "**With issues**: {with_issues}");

    if with_issues == 0 {
        let _ = writeln!(out, "\nAll tags are valid");
    } else {
        let _ = writeln!(out);
        for (tag, issues, _suggestion, _) in results {
            if !issues.is_empty() {
                let _ = writeln!(out, "### `{tag}`");
                for issue in issues {
                    let _ = writeln!(out, "- {issue}");
                }
            }
        }
    }

    out
}
