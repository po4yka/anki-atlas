use std::path::Path;

use indexer::embeddings::{EmbeddingInput, EmbeddingPart, EmbeddingTask};
use indexer::service::{ChunkForIndexing, MultimodalNoteForIndexing, NoteForIndexing};
use sha2::{Digest, Sha256};

use super::media::extract_media_refs;
use super::service::NoteIndexRow;
use crate::error::SurfaceError;

pub(super) fn short_preview_label(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "asset".to_string();
    }
    let preview = trimmed.chars().take(80).collect::<String>();
    preview.replace('\n', " ")
}

pub(super) fn build_chunk_id(note_id: i64, chunk_kind: &str, suffix: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(note_id.to_le_bytes());
    hasher.update(chunk_kind.as_bytes());
    hasher.update(suffix.as_bytes());
    format!(
        "{note_id}:{chunk_kind}:{}",
        hex::encode(&hasher.finalize()[..6])
    )
}

pub(super) fn build_note_chunks(
    row: &NoteIndexRow,
    media_root: Option<&Path>,
) -> Result<MultimodalNoteForIndexing, SurfaceError> {
    let mut chunks = vec![ChunkForIndexing {
        chunk_id: format!("{}:text_primary", row.note_id),
        chunk_kind: "text_primary".to_string(),
        modality: "text".to_string(),
        embedding_input: EmbeddingInput::text_with_task(
            row.normalized_text.clone(),
            EmbeddingTask::RetrievalDocument,
        ),
        sparse_text: Some(row.normalized_text.clone()),
        source_field: None,
        asset_rel_path: None,
        mime_type: Some("text/plain".to_string()),
        preview_label: Some(short_preview_label(&row.normalized_text)),
        hash_component: row.normalized_text.clone(),
    }];

    if let Some(media_root) = media_root {
        for asset in extract_media_refs(&row.fields_json, row.raw_fields.as_deref()) {
            let asset_path = media_root.join(&asset.asset_rel_path);
            if !asset_path.exists() {
                continue;
            }

            let bytes = std::fs::read(&asset_path)?;
            let digest = {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                hex::encode(&hasher.finalize()[..8])
            };
            let suffix = format!(
                "{}:{}",
                asset.source_field.as_deref().unwrap_or("asset"),
                asset.asset_rel_path
            );
            chunks.push(ChunkForIndexing {
                chunk_id: build_chunk_id(row.note_id, "asset", &suffix),
                chunk_kind: "asset".to_string(),
                modality: asset.modality.clone(),
                embedding_input: EmbeddingInput {
                    parts: vec![EmbeddingPart::InlineBytes {
                        mime_type: asset.mime_type.clone(),
                        data: bytes,
                        display_name: Some(asset.preview_label.clone()),
                    }],
                    task: EmbeddingTask::RetrievalDocument,
                    title: Some(asset.preview_label.clone()),
                    output_dimensionality: None,
                },
                sparse_text: None,
                source_field: asset.source_field.clone(),
                asset_rel_path: Some(asset.asset_rel_path.clone()),
                mime_type: Some(asset.mime_type.clone()),
                preview_label: Some(asset.preview_label.clone()),
                hash_component: digest,
            });
        }
    }

    Ok(MultimodalNoteForIndexing {
        note: NoteForIndexing {
            note_id: row.note_id,
            model_id: row.model_id,
            normalized_text: row.normalized_text.clone(),
            tags: row.tags.clone(),
            deck_names: row.deck_names.clone(),
            mature: row.mature,
            lapses: row.lapses,
            reps: row.reps,
            fail_rate: row.fail_rate,
        },
        chunks,
    })
}
