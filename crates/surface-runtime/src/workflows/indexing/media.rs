use std::collections::HashSet;
use std::path::Path;

use regex::Regex;

fn sanitize_relative_asset_path(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty()
        || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("data:")
        || trimmed.starts_with('/')
    {
        return None;
    }
    let candidate = trimmed.replace('\\', "/");
    if candidate.split('/').any(|part| part == "..") {
        return None;
    }
    Some(candidate)
}

pub(super) fn detect_media_type(rel_path: &str) -> Option<(&'static str, &'static str)> {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())?;

    match ext.as_str() {
        "png" => Some(("image", "image/png")),
        "jpg" | "jpeg" => Some(("image", "image/jpeg")),
        "gif" => Some(("image", "image/gif")),
        "webp" => Some(("image", "image/webp")),
        "mp3" => Some(("audio", "audio/mpeg")),
        "wav" => Some(("audio", "audio/wav")),
        "ogg" => Some(("audio", "audio/ogg")),
        "m4a" => Some(("audio", "audio/mp4")),
        "aac" => Some(("audio", "audio/aac")),
        "flac" => Some(("audio", "audio/flac")),
        "mp4" => Some(("video", "video/mp4")),
        "mov" => Some(("video", "video/quicktime")),
        "webm" => Some(("video", "video/webm")),
        "pdf" => Some(("document", "application/pdf")),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MediaAssetRef {
    pub(super) source_field: Option<String>,
    pub(super) asset_rel_path: String,
    pub(super) mime_type: String,
    pub(super) modality: String,
    pub(super) preview_label: String,
}

pub(super) fn capture_asset_refs(
    content: &str,
    source_field: Option<&str>,
    sink: &mut Vec<MediaAssetRef>,
    seen: &mut HashSet<(Option<String>, String)>,
) {
    let source_field = source_field.map(ToString::to_string);
    let patterns = [
        r#"(?i)<img[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<video[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<source[^>]+src=["']?([^"' >]+)"#,
        r#"(?i)<a[^>]+href=["']?([^"' >]+\.pdf(?:\?[^"' >]*)?)"#,
        r#"(?i)<embed[^>]+src=["']?([^"' >]+\.pdf(?:\?[^"' >]*)?)"#,
    ];

    for pattern in patterns {
        let Ok(regex) = Regex::new(pattern) else {
            continue;
        };
        for captures in regex.captures_iter(content) {
            let Some(path_match) = captures.get(1) else {
                continue;
            };
            let Some(asset_rel_path) = sanitize_relative_asset_path(path_match.as_str()) else {
                continue;
            };
            let Some((modality, mime_type)) = detect_media_type(&asset_rel_path) else {
                continue;
            };
            let key = (source_field.clone(), asset_rel_path.clone());
            if !seen.insert(key) {
                continue;
            }
            sink.push(MediaAssetRef {
                source_field: source_field.clone(),
                preview_label: Path::new(&asset_rel_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| asset_rel_path.clone()),
                asset_rel_path,
                mime_type: mime_type.to_string(),
                modality: modality.to_string(),
            });
        }
    }

    let Ok(sound_regex) = Regex::new(r#"\[sound:([^\]]+)\]"#) else {
        return;
    };
    for captures in sound_regex.captures_iter(content) {
        let Some(path_match) = captures.get(1) else {
            continue;
        };
        let Some(asset_rel_path) = sanitize_relative_asset_path(path_match.as_str()) else {
            continue;
        };
        let Some((modality, mime_type)) = detect_media_type(&asset_rel_path) else {
            continue;
        };
        let key = (source_field.clone(), asset_rel_path.clone());
        if !seen.insert(key) {
            continue;
        }
        sink.push(MediaAssetRef {
            source_field: source_field.clone(),
            preview_label: Path::new(&asset_rel_path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| asset_rel_path.clone()),
            asset_rel_path,
            mime_type: mime_type.to_string(),
            modality: modality.to_string(),
        });
    }
}

pub(super) fn extract_media_refs(
    fields_json: &serde_json::Value,
    raw_fields: Option<&str>,
) -> Vec<MediaAssetRef> {
    let mut refs = Vec::new();
    let mut seen = std::collections::HashSet::new();

    if let Some(object) = fields_json.as_object() {
        for (field_name, value) in object {
            if let Some(content) = value.as_str() {
                capture_asset_refs(content, Some(field_name), &mut refs, &mut seen);
            }
        }
    }

    if let Some(raw_fields) = raw_fields {
        for raw_field in raw_fields.split('\u{1f}') {
            capture_asset_refs(raw_field, None, &mut refs, &mut seen);
        }
    }

    refs
}
