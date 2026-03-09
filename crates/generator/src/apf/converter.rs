/// Convert Markdown content to HTML.
/// Uses a basic markdown-to-HTML converter with code highlighting support.
pub fn markdown_to_html(md_content: &str, sanitize: bool) -> String {
    let _ = (md_content, sanitize);
    todo!()
}

/// Sanitize HTML using ammonia with Anki-safe tag/attribute allowlist.
pub fn sanitize_html(html: &str) -> String {
    let _ = html;
    todo!()
}

/// Convert a single APF field from Markdown to HTML.
pub fn convert_apf_field(field_content: &str) -> String {
    let _ = field_content;
    todo!()
}

/// Highlight code with language class annotations.
pub fn highlight_code(code: &str, language: Option<&str>) -> String {
    let _ = (code, language);
    todo!()
}
