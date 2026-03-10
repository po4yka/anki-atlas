use rmcp::model::{CallToolResult, Content};
use serde::Serialize;

use crate::tools::{OutputMode, ToolError};

pub fn success_result<T: Serialize>(
    output_mode: OutputMode,
    markdown: String,
    data: &T,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let structured = serde_json::to_value(data)
        .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?;
    let text = match output_mode {
        OutputMode::Markdown => markdown,
        OutputMode::Json => serde_json::to_string_pretty(&structured)
            .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?,
    };
    let mut result = CallToolResult::structured(structured);
    result.content = vec![Content::text(text)];
    Ok(result)
}

pub fn error_result(
    output_mode: OutputMode,
    error: ToolError,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let structured = serde_json::to_value(&error).map_err(|serialization| {
        rmcp::ErrorData::internal_error(serialization.to_string(), None)
    })?;
    let text = match output_mode {
        OutputMode::Markdown => {
            let mut out = format!("## Error\n\n**{}**\n\n{}", error.error, error.message);
            if let Some(details) = &error.details {
                out.push_str(&format!("\n\n{details}"));
            }
            out
        }
        OutputMode::Json => serde_json::to_string_pretty(&structured).map_err(|serialization| {
            rmcp::ErrorData::internal_error(serialization.to_string(), None)
        })?,
    };
    let mut result = CallToolResult::structured_error(structured);
    result.content = vec![Content::text(text)];
    Ok(result)
}
