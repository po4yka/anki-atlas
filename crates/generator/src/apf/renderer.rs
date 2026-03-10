use serde::{Deserialize, Serialize};

pub const PROMPT_VERSION: &str = "apf-v2.1";

/// Input spec for rendering a card to APF HTML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardSpec {
    pub card_index: u32,
    pub slug: String,
    pub slug_base: Option<String>,
    pub lang: String,
    pub card_type: String,
    pub tags: Vec<String>,
    pub guid: String,
    pub source_path: Option<String>,
    pub source_anchor: Option<String>,
    pub title: String,
    pub key_point_code: Option<String>,
    pub key_point_code_lang: Option<String>,
    pub key_point_notes: Vec<String>,
    pub other_notes: Option<String>,
    pub extra: Option<String>,
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn build_manifest(spec: &CardSpec) -> String {
    let slug_base = spec
        .slug_base
        .clone()
        .unwrap_or_else(|| match spec.slug.rfind('-') {
            Some(pos) => spec.slug[..pos].to_string(),
            None => spec.slug.clone(),
        });

    let tags_json: Vec<String> = spec.tags.iter().map(|t| format!("\"{t}\"")).collect();

    let mut parts = vec![
        format!("\"slug\":\"{}\"", spec.slug),
        format!("\"slug_base\":\"{}\"", slug_base),
        format!("\"lang\":\"{}\"", spec.lang),
        format!("\"type\":\"{}\"", spec.card_type),
        format!("\"tags\":[{}]", tags_json.join(",")),
    ];

    if let Some(ref path) = spec.source_path {
        parts.push(format!("\"source_path\":\"{path}\""));
    }
    if let Some(ref anchor) = spec.source_anchor {
        parts.push(format!("\"source_anchor\":\"{anchor}\""));
    }

    format!("{{{}}}", parts.join(","))
}

/// Render a single card spec to APF HTML.
pub fn render(spec: &CardSpec) -> String {
    let tags_str = spec.tags.join(" ");
    let mut out = String::new();

    // Header sentinels
    out.push_str(&format!("<!-- PROMPT_VERSION: {PROMPT_VERSION} -->\n"));
    out.push_str("<!-- BEGIN_CARDS -->\n");
    out.push_str(&format!(
        "<!-- Card {} | slug: {} | CardType: {} | Tags: {} -->\n",
        spec.card_index, spec.slug, spec.card_type, tags_str
    ));

    // Title
    out.push_str(&format!("<!-- Title -->\n{}\n", spec.title));

    // Key point code block
    out.push_str("<!-- Key point (code block / image) -->\n");
    if let Some(ref code) = spec.key_point_code {
        let lang = spec.key_point_code_lang.as_deref().unwrap_or("plaintext");
        out.push_str(&format!(
            "<pre><code class=\"language-{}\">{}</code></pre>\n",
            lang,
            escape_html(code)
        ));
    }

    // Key point notes
    out.push_str("<!-- Key point notes -->\n");
    if spec.key_point_notes.is_empty() {
        out.push_str("<ul></ul>\n");
    } else {
        out.push_str("<ul>");
        for note in &spec.key_point_notes {
            out.push_str(&format!("<li>{note}</li>"));
        }
        out.push_str("</ul>\n");
    }

    // Other notes
    out.push_str("<!-- Other notes -->\n");
    if let Some(ref notes) = spec.other_notes {
        out.push_str(&format!("{notes}\n"));
    }

    // Extra
    out.push_str("<!-- Extra -->\n");
    if let Some(ref extra) = spec.extra {
        out.push_str(&format!("{extra}\n"));
    }

    // Manifest
    out.push_str(&format!("<!-- manifest:{} -->\n", build_manifest(spec)));

    // Footer sentinels
    out.push_str("<!-- END_CARDS -->\n");
    out.push_str("END_OF_CARDS");

    out
}

/// Render multiple card specs, separated by `<!-- CARD_SEPARATOR -->`.
pub fn render_batch(specs: &[CardSpec]) -> String {
    if specs.is_empty() {
        return String::new();
    }

    specs
        .iter()
        .map(render)
        .collect::<Vec<_>>()
        .join("\n<!-- CARD_SEPARATOR -->\n")
}
