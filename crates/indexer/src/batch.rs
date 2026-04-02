use std::collections::HashMap;

use crate::embeddings::{self, EmbeddingError, EmbeddingInput, EmbeddingProvider};
use crate::qdrant::{NotePayload, QdrantRepository, SparseVector};
use crate::service::emit_progress;
use crate::service::{
    ChunkForIndexing, IndexProgressCallback, IndexProgressStage, MultimodalNoteForIndexing,
};

/// Result of the diff stage: notes that need (re-)embedding.
pub(crate) struct NoteToEmbed<'a> {
    pub note: &'a MultimodalNoteForIndexing,
    pub new_hash: String,
    pub chunks: Vec<ChunkForIndexing>,
}

/// Compute which notes need re-embedding by comparing content hashes.
///
/// Returns `(to_embed, skipped_count)`.
pub(crate) fn compute_changed_notes<'a>(
    notes: &'a [MultimodalNoteForIndexing],
    existing_hashes: &HashMap<i64, String>,
    model_name: &str,
    force_reindex: bool,
    progress: Option<&IndexProgressCallback>,
) -> (Vec<NoteToEmbed<'a>>, usize) {
    let mut to_embed = Vec::new();
    let mut skipped = 0usize;

    for (idx, note) in notes.iter().enumerate() {
        let chunks = note_effective_chunks(note);
        let new_hash = embeddings::content_hash_parts(
            model_name,
            chunks.iter().map(|chunk| chunk.hash_component.as_str()),
        );
        if !force_reindex {
            if let Some(existing) = existing_hashes.get(&note.note.note_id) {
                if *existing == new_hash {
                    skipped += 1;
                    emit_progress(
                        progress,
                        IndexProgressStage::Diffing,
                        idx + 1,
                        notes.len(),
                        format!("skipped unchanged note {}", note.note.note_id),
                    );
                    continue;
                }
            }
        }
        to_embed.push(NoteToEmbed {
            note,
            new_hash,
            chunks,
        });
        emit_progress(
            progress,
            IndexProgressStage::Diffing,
            idx + 1,
            notes.len(),
            format!("queued note {} for embedding", note.note.note_id),
        );
    }

    (to_embed, skipped)
}

/// Embed all chunks from the notes-to-embed list.
///
/// Returns flat `Vec<Vec<f32>>` in the same order as the chunks.
pub(crate) async fn batch_embed_chunks<E: EmbeddingProvider>(
    embedding: &E,
    to_embed: &[NoteToEmbed<'_>],
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let inputs: Vec<EmbeddingInput> = to_embed
        .iter()
        .flat_map(|e| &e.chunks)
        .map(|chunk| chunk.embedding_input.clone())
        .collect();
    embedding.embed_inputs(&inputs).await
}

/// Build the upsert payloads (one per chunk) from the notes-to-embed list.
pub(crate) fn build_upsert_payloads(to_embed: &[NoteToEmbed<'_>]) -> Vec<NotePayload> {
    to_embed
        .iter()
        .flat_map(|e| {
            e.chunks.iter().map(move |chunk| NotePayload {
                note_id: e.note.note.note_id,
                model_id: e.note.note.model_id,
                deck_names: e.note.note.deck_names.clone(),
                tags: e.note.note.tags.clone(),
                content_hash: e.new_hash.clone(),
                mature: e.note.note.mature,
                lapses: e.note.note.lapses,
                reps: e.note.note.reps,
                fail_rate: e.note.note.fail_rate,
                chunk_id: chunk.chunk_id.clone(),
                chunk_kind: chunk.chunk_kind.clone(),
                modality: chunk.modality.clone(),
                source_field: chunk.source_field.clone(),
                asset_rel_path: chunk.asset_rel_path.clone(),
                mime_type: chunk.mime_type.clone(),
                preview_label: chunk.preview_label.clone(),
            })
        })
        .collect()
}

/// Generate sparse vectors from the chunks' `sparse_text` field.
pub(crate) fn generate_sparse_vectors(to_embed: &[NoteToEmbed<'_>]) -> Vec<SparseVector> {
    to_embed
        .iter()
        .flat_map(|e| &e.chunks)
        .map(|chunk| {
            chunk
                .sparse_text
                .as_deref()
                .map(QdrantRepository::text_to_sparse_vector)
                .unwrap_or_default()
        })
        .collect()
}

/// Return the effective chunks for a note: explicit chunks if present,
/// otherwise a single default text chunk.
pub(crate) fn note_effective_chunks(note: &MultimodalNoteForIndexing) -> Vec<ChunkForIndexing> {
    use crate::service::default_text_chunk;
    if note.chunks.is_empty() {
        vec![default_text_chunk(
            &note.note.normalized_text,
            note.note.note_id,
        )]
    } else {
        note.chunks.clone()
    }
}
