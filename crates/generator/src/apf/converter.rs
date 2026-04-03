use std::collections::HashSet;

use ammonia::Builder;
use pulldown_cmark::{Options, Parser, html};

/// Controls HTML sanitization behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlSanitization {
    Sanitize,
    Raw,
}

/// Convert Markdown content to HTML.
/// Uses pulldown-cmark for conversion and optionally sanitizes with ammonia.
pub fn markdown_to_html(md_content: &str, sanitize: HtmlSanitization) -> String {
    if md_content.is_empty() || md_content.trim().is_empty() {
        return String::new();
    }

    let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(md_content, options);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);

    let result = html_output.trim().to_string();

    if sanitize == HtmlSanitization::Sanitize {
        sanitize_html(&result)
    } else {
        result
    }
}

/// Sanitize HTML using ammonia with Anki-safe tag/attribute allowlist.
pub fn sanitize_html(html: &str) -> String {
    if html.is_empty() {
        return String::new();
    }

    let tags: HashSet<&str> = [
        "p",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "s",
        "code",
        "pre",
        "br",
        "hr",
        "div",
        "span",
        "ul",
        "ol",
        "li",
        "a",
        "img",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "sub",
        "sup",
        "dl",
        "dt",
        "dd",
    ]
    .into_iter()
    .collect();

    let tag_attributes: std::collections::HashMap<&str, HashSet<&str>> = [
        ("a", ["href", "title", "class"].into_iter().collect()),
        (
            "img",
            ["src", "alt", "title", "width", "height", "class"]
                .into_iter()
                .collect(),
        ),
        ("code", ["class"].into_iter().collect()),
        ("pre", ["class"].into_iter().collect()),
        ("div", ["class"].into_iter().collect()),
        ("span", ["class"].into_iter().collect()),
        ("td", ["colspan", "rowspan", "class"].into_iter().collect()),
        ("th", ["colspan", "rowspan", "class"].into_iter().collect()),
        ("table", ["class"].into_iter().collect()),
        ("p", ["class"].into_iter().collect()),
    ]
    .into_iter()
    .collect();

    Builder::new()
        .tags(tags)
        .tag_attributes(tag_attributes)
        .clean(html)
        .to_string()
}

/// Convert a single APF field from Markdown to HTML.
/// Runs markdown conversion followed by sanitization.
pub fn convert_apf_field(field_content: &str) -> String {
    if field_content.is_empty() {
        return String::new();
    }
    markdown_to_html(field_content, HtmlSanitization::Sanitize)
}

/// Highlight code with language class annotations.
/// Wraps code in `<pre><code class="language-{lang}">` with HTML escaping.
pub fn highlight_code(code: &str, language: Option<&str>) -> String {
    let lang = language.unwrap_or("text");
    let escaped = html_escape::encode_text(code);
    format!("<pre><code class=\"language-{lang}\">{escaped}</code></pre>")
}
