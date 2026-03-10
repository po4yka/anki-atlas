use anki_atlas_mcp::handlers::{error_result, success_result};
use anki_atlas_mcp::tools::{OutputMode, ToolError};

#[test]
fn success_result_preserves_structured_content() {
    let result = success_result(
        OutputMode::Markdown,
        "markdown".to_string(),
        &serde_json::json!({"ok": true}),
    )
    .expect("success result");
    assert_eq!(result.is_error, Some(false));
    assert_eq!(
        result.structured_content,
        Some(serde_json::json!({"ok": true}))
    );
}

#[test]
fn error_result_marks_error_and_formats_markdown() {
    let result = error_result(
        OutputMode::Markdown,
        ToolError {
            error: "invalid_input".to_string(),
            message: "bad input".to_string(),
            details: Some("details".to_string()),
        },
    )
    .expect("error result");
    assert_eq!(result.is_error, Some(true));
    let text = result
        .content
        .first()
        .and_then(|content| content.as_text())
        .map(|text| text.text.clone())
        .expect("text content");
    assert!(text.contains("bad input"));
}
