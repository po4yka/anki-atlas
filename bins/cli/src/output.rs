use surface_contracts::analytics::{
    DuplicateCluster, DuplicateStats, LabelingStats, TopicCoverage, TopicGap, WeakNote,
};
use surface_contracts::search::{ChunkSearchResponse, SearchResponse};
use surface_runtime::{
    GeneratePreview, IndexExecutionSummary, ObsidianScanPreview, SyncExecutionSummary,
    TagAuditSummary, ValidationSummary,
};

pub fn print_search_result(result: &SearchResponse, verbose: bool) {
    println!("query: {}", result.query);
    println!("results: {}", result.results.len());
    for item in &result.results {
        let headline = item.headline.as_deref().unwrap_or("(no headline)");
        println!(
            "- note={} score={:.4} sources={} headline={}",
            item.note_id,
            item.rrf_score,
            item.sources.join(","),
            headline
        );
        if verbose {
            println!(
                "  semantic={:?} fts={:?} rerank={:?} match_modality={:?} match_chunk_kind={:?} match_source_field={:?} match_asset_rel_path={:?}",
                item.semantic_score,
                item.fts_score,
                item.rerank_score,
                item.match_modality,
                item.match_chunk_kind,
                item.match_source_field,
                item.match_asset_rel_path
            );
        }
    }
    if verbose {
        println!(
            "lexical_mode={:?} fallback={} rerank_applied={} rerank_model={:?} rerank_top_n={:?}",
            result.lexical_mode,
            result.lexical_fallback_used,
            result.rerank_applied,
            result.rerank_model,
            result.rerank_top_n
        );
        if !result.query_suggestions.is_empty() {
            println!("query_suggestions: {}", result.query_suggestions.join(", "));
        }
        if !result.autocomplete_suggestions.is_empty() {
            println!(
                "autocomplete_suggestions: {}",
                result.autocomplete_suggestions.join(", ")
            );
        }
    }
}

pub fn print_chunk_search_result(result: &ChunkSearchResponse, verbose: bool) {
    println!("query: {}", result.query);
    println!("results: {}", result.results.len());
    for item in &result.results {
        println!(
            "- note={} chunk={} kind={} modality={} score={:.4}",
            item.note_id, item.chunk_id, item.chunk_kind, item.modality, item.score
        );
        if verbose {
            println!(
                "  source_field={:?} asset_rel_path={:?} mime_type={:?} preview_label={:?}",
                item.source_field, item.asset_rel_path, item.mime_type, item.preview_label
            );
        }
    }
}

pub fn print_topics_tree(topics: &[serde_json::Value]) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(topics)?);
    Ok(())
}

pub fn print_taxonomy_load(topic_count: usize, root_count: usize) {
    println!("taxonomy loaded");
    println!("topics: {topic_count}");
    println!("roots: {root_count}");
}

pub fn print_labeling_summary(stats: &LabelingStats) {
    println!("labeling complete");
    println!("notes_processed: {}", stats.notes_processed);
    println!("assignments_created: {}", stats.assignments_created);
    println!("topics_matched: {}", stats.topics_matched);
}

pub fn print_coverage(coverage: &TopicCoverage) {
    println!("topic: {} ({})", coverage.path, coverage.label);
    println!("note_count: {}", coverage.note_count);
    println!("subtree_count: {}", coverage.subtree_count);
    println!(
        "covered_children: {}/{}",
        coverage.covered_children, coverage.child_count
    );
    println!("mature_count: {}", coverage.mature_count);
    println!("avg_confidence: {:.3}", coverage.avg_confidence);
    println!("weak_notes: {}", coverage.weak_notes);
    println!("avg_lapses: {:.3}", coverage.avg_lapses);
}

pub fn print_gaps(gaps: &[TopicGap]) {
    println!("gaps: {}", gaps.len());
    for gap in gaps {
        println!(
            "- {} [{}] notes={} threshold={}",
            gap.path,
            serde_json::to_string(&gap.gap_type).unwrap_or_else(|_| "\"unknown\"".to_string()),
            gap.note_count,
            gap.threshold
        );
    }
}

pub fn print_weak_notes(notes: &[WeakNote]) {
    println!("weak_notes: {}", notes.len());
    for note in notes {
        println!(
            "- note={} confidence={:.3} lapses={} fail_rate={:?}",
            note.note_id, note.confidence, note.lapses, note.fail_rate
        );
        println!("  {}", note.normalized_text);
    }
}

pub fn print_duplicates(clusters: &[DuplicateCluster], stats: &DuplicateStats, verbose: bool) {
    println!("clusters: {}", stats.clusters_found);
    println!("notes_scanned: {}", stats.notes_scanned);
    println!("total_duplicates: {}", stats.total_duplicates);
    println!("avg_cluster_size: {:.2}", stats.avg_cluster_size);
    for cluster in clusters {
        println!(
            "- representative={} size={} decks={} tags={}",
            cluster.representative_id,
            cluster.size(),
            cluster.deck_names.join(","),
            cluster.tags.join(",")
        );
        if verbose {
            println!("  {}", cluster.representative_text);
            for duplicate in &cluster.duplicates {
                println!(
                    "  duplicate note={} similarity={:.4} decks={} tags={}",
                    duplicate.note_id,
                    duplicate.similarity,
                    duplicate.deck_names.join(","),
                    duplicate.tags.join(",")
                );
            }
        }
    }
}

pub fn print_generate_preview(preview: &GeneratePreview) {
    println!("source: {}", preview.source_file.display());
    println!(
        "title: {}",
        preview.title.as_deref().unwrap_or("(untitled)")
    );
    println!("estimated_cards: {}", preview.estimated_cards);
    println!("sections: {}", preview.sections.join(", "));
    if !preview.warnings.is_empty() {
        println!("warnings: {}", preview.warnings.join(" | "));
    }
}

pub fn print_validation(summary: &ValidationSummary) {
    println!("source: {}", summary.source_file.display());
    println!("valid: {}", summary.is_valid);
    println!("issues: {}", summary.issues.len());
    for issue in &summary.issues {
        println!(
            "- {} [{}] {}",
            issue.severity, issue.location, issue.message
        );
    }
    if let Some(quality) = &summary.quality {
        println!(
            "quality: overall={:.3} clarity={:.3} atomicity={:.3} testability={:.3} memorability={:.3} accuracy={:.3}",
            quality.overall(),
            quality.clarity,
            quality.atomicity,
            quality.testability,
            quality.memorability,
            quality.accuracy
        );
    }
}

pub fn print_obsidian_scan(preview: &ObsidianScanPreview) {
    println!("vault: {}", preview.vault_path.display());
    println!("notes: {}", preview.note_count);
    println!("generated_cards: {}", preview.generated_cards);
    println!("orphaned_notes: {}", preview.orphaned_notes.len());
    println!("broken_links: {}", preview.broken_links.len());
    for note in preview.notes.iter().take(10) {
        println!(
            "- {} title={} sections={}",
            note.path.display(),
            note.title.as_deref().unwrap_or("(untitled)"),
            note.sections
        );
    }
}

pub fn print_tag_audit(summary: &TagAuditSummary) {
    println!("source: {}", summary.source_file.display());
    println!("applied_fixes: {}", summary.applied_fixes);
    for entry in &summary.entries {
        println!(
            "- {} valid={} normalized={} suggestion={}",
            entry.tag,
            entry.valid,
            entry.normalized,
            entry.suggestion.as_deref().unwrap_or("-")
        );
        for issue in &entry.issues {
            println!("  issue: {issue}");
        }
    }
}

pub fn print_sync_summary(summary: &SyncExecutionSummary) {
    println!("source: {}", summary.source.display());
    println!("migrations_applied: {}", summary.migrations_applied);
    println!("decks_upserted: {}", summary.sync.decks_upserted);
    println!("models_upserted: {}", summary.sync.models_upserted);
    println!("notes_upserted: {}", summary.sync.notes_upserted);
    println!("notes_deleted: {}", summary.sync.notes_deleted);
    println!("cards_upserted: {}", summary.sync.cards_upserted);
    println!("card_stats_upserted: {}", summary.sync.card_stats_upserted);
    println!("duration_ms: {}", summary.sync.duration_ms);
    if let Some(index) = &summary.index {
        print_index_summary(index);
    }
}

pub fn print_index_summary(summary: &IndexExecutionSummary) {
    println!("index_reindex_mode: {}", summary.reindex_mode);
    println!("notes_processed: {}", summary.stats.notes_processed);
    println!("notes_embedded: {}", summary.stats.notes_embedded);
    println!("notes_skipped: {}", summary.stats.notes_skipped);
    println!("notes_deleted: {}", summary.stats.notes_deleted);
    if !summary.stats.errors.is_empty() {
        println!("errors: {}", summary.stats.errors.join(" | "));
    }
}
