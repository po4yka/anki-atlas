use std::path::Path;

use card::CardRegistry;
use cardloop::{
    CardloopStore, ClusterBuilder, ItemStatus, LoopKind, ProgressionLog, QueueBuilder,
    models::ProgressionEvent,
    scanners::{Scanner, audit::AuditScanner},
};
use chrono::Utc;
use validation::{
    ContentValidator, FormatValidator, HtmlValidator, RelevanceValidator, TagValidator,
    ValidationPipeline,
};

use crate::args::{CardloopArgs, CardloopCommand};

/// Default directory for cardloop state.
const CARDLOOP_DIR: &str = ".cardloop";

fn store_path() -> String {
    format!("{CARDLOOP_DIR}/state.db")
}

fn progression_path() -> std::path::PathBuf {
    Path::new(CARDLOOP_DIR).join("progression.jsonl")
}

fn default_pipeline() -> ValidationPipeline {
    ValidationPipeline::new(vec![
        Box::new(ContentValidator::default()),
        Box::new(FormatValidator),
        Box::new(HtmlValidator),
        Box::new(TagValidator::default()),
        Box::new(RelevanceValidator::new()),
    ])
}

pub fn run(args: &CardloopArgs) -> anyhow::Result<()> {
    match &args.command {
        CardloopCommand::Scan(scan_args) => cmd_scan(scan_args),
        CardloopCommand::Status(status_args) => cmd_status(status_args),
        CardloopCommand::Next(next_args) => cmd_next(next_args),
        CardloopCommand::Resolve(resolve_args) => cmd_resolve(resolve_args),
        CardloopCommand::Log(log_args) => cmd_log(log_args),
    }
}

fn cmd_scan(args: &crate::args::CardloopScanArgs) -> anyhow::Result<()> {
    let store = CardloopStore::open(&store_path())?;
    let log = ProgressionLog::open(&progression_path())?;

    let scores_before = store.compute_scores()?;
    let scan_number = store.increment_scan_count()?;

    let loop_kind = match args.loop_kind.as_str() {
        "audit" => Some(LoopKind::Audit),
        "generation" => Some(LoopKind::Generation),
        "all" => None,
        other => anyhow::bail!("unknown loop kind: {other} (expected: audit, generation, all)"),
    };

    let mut all_items = Vec::new();

    // Run audit scanner
    if loop_kind.is_none() || loop_kind == Some(LoopKind::Audit) {
        let registry_path = args.registry.to_str().unwrap_or("");
        let registry = CardRegistry::open(registry_path)?;
        let pipeline = default_pipeline();
        let scanner = AuditScanner::new(&registry, &pipeline);
        let items = scanner.scan(scan_number)?;
        all_items.extend(items);
    }

    // TODO: Run generation scanner when implemented

    // Assign cluster IDs before persisting.
    let all_items = ClusterBuilder::assign(all_items);

    let new_count = all_items.len();

    // Collect IDs for reconciliation
    let current_ids: std::collections::HashSet<String> =
        all_items.iter().map(|i| i.id.clone()).collect();

    store.upsert_items(&all_items)?;
    let stale_resolved = store.reconcile_stale(&current_ids)?;

    let scores_after = store.compute_scores()?;

    // Log progression
    let item_ids: Vec<String> = all_items.iter().map(|i| i.id.clone()).collect();
    log.append(&ProgressionEvent {
        timestamp: Utc::now(),
        action: "scan".into(),
        item_ids,
        actor: "agent".into(),
        note: Some(format!(
            "scan #{scan_number}: {new_count} items found, {stale_resolved} auto-resolved"
        )),
        scores_before: Some(scores_before),
        scores_after: Some(scores_after.clone()),
    })?;

    // Print summary
    println!("Scan #{scan_number} complete");
    println!("  Items found:    {new_count}");
    println!("  Auto-resolved:  {stale_resolved}");
    println!("  Open:           {}", scores_after.open_count);
    println!("  Fixed:          {}", scores_after.fixed_count);
    println!("  Score:          {:.1}%", scores_after.overall * 100.0);

    Ok(())
}

fn cmd_status(args: &crate::args::CardloopStatusArgs) -> anyhow::Result<()> {
    let store = CardloopStore::open(&store_path())?;
    let scores = store.compute_scores()?;
    let scan_count = store.scan_count()?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&scores)?);
        return Ok(());
    }

    println!("=== Cardloop Status ===");
    println!("Scans:         {scan_count}");
    println!("Score:         {:.1}%", scores.overall * 100.0);
    println!();
    println!("Open:          {}", scores.open_count);
    println!("Fixed:         {}", scores.fixed_count);
    println!("Skipped:       {}", scores.skipped_count);
    println!();
    println!("By tier (open):");
    println!("  T1 AutoFix:  {}", scores.by_tier[0]);
    println!("  T2 QuickFix: {}", scores.by_tier[1]);
    println!("  T3 Rework:   {}", scores.by_tier[2]);
    println!("  T4 Delete:   {}", scores.by_tier[3]);
    println!();
    println!("By loop (open):");
    println!("  Generation:  {}", scores.by_loop.0);
    println!("  Audit:       {}", scores.by_loop.1);

    Ok(())
}

fn cmd_next(args: &crate::args::CardloopNextArgs) -> anyhow::Result<()> {
    let store = CardloopStore::open(&store_path())?;

    let loop_kind = args
        .loop_kind
        .as_deref()
        .map(|s| {
            s.parse::<LoopKind>()
                .map_err(|_| anyhow::anyhow!("unknown loop kind: {s}"))
        })
        .transpose()?;

    let items =
        QueueBuilder::build_filtered(&store, loop_kind, args.count, args.cluster.as_deref())?;

    if items.is_empty() {
        println!("Queue is empty. Run `cardloop scan` to populate.");
        return Ok(());
    }

    for (i, item) in items.iter().enumerate() {
        println!("--- Item {}/{} ---", i + 1, items.len());
        println!("ID:          {}", item.id);
        println!("Loop:        {}", item.loop_kind.as_str());
        println!("Tier:        {:?}", item.tier);
        println!("Slug:        {}", item.slug.as_deref().unwrap_or("-"));
        println!("Source:      {}", item.source_path);
        println!("Summary:     {}", item.summary);
        if let Some(detail) = &item.detail {
            println!("Detail:      {detail}");
        }
        println!(
            "Resolve:     anki-atlas cardloop resolve {} --attest \"...\"",
            &item.id[..8.min(item.id.len())]
        );
        println!();
    }

    Ok(())
}

fn cmd_resolve(args: &crate::args::CardloopResolveArgs) -> anyhow::Result<()> {
    let store = CardloopStore::open(&store_path())?;
    let log = ProgressionLog::open(&progression_path())?;

    let new_status: ItemStatus = args.status.parse().map_err(|_| {
        anyhow::anyhow!(
            "unknown status: {} (expected: fixed, skipped, wontfix)",
            args.status
        )
    })?;

    // Try prefix match first, then exact match
    let item = store
        .get_item_by_prefix(&args.id)?
        .or_else(|| store.get_item(&args.id).ok().flatten())
        .ok_or_else(|| anyhow::anyhow!("no work item matching '{}'", args.id))?;

    let scores_before = store.compute_scores()?;
    let updated = store.transition(&item.id, new_status, args.attest.as_deref())?;

    // Verification gate: if resolving as Fixed and we have a registry, re-run audit
    // on this card to confirm the issue is actually gone. If the same issue is still
    // present, reopen the item automatically.
    if new_status == ItemStatus::Fixed {
        if let (Some(slug), Some(registry_path)) = (&updated.slug, &args.registry) {
            let registry = CardRegistry::open(registry_path.to_str().unwrap_or(""))?;
            let pipeline = default_pipeline();
            let scanner = AuditScanner::new(&registry, &pipeline);
            // Use scan_number=0 as a probe — we only care about IDs, not persisting.
            let probe_items = scanner.scan(0)?;
            let still_present = probe_items
                .iter()
                .any(|i| i.id == updated.id && i.slug.as_deref() == Some(slug.as_str()));
            if still_present {
                // Reopen: transition back to Open
                store.transition(
                    &updated.id,
                    ItemStatus::Open,
                    Some("auto-reopened: issue still detected after resolve"),
                )?;
                eprintln!(
                    "WARNING: Issue still detected for '{}' after resolve. Item {} reopened.",
                    slug,
                    &updated.id[..8.min(updated.id.len())]
                );
                return Ok(());
            }
        }
    }

    let scores_after = store.compute_scores()?;

    log.append(&ProgressionEvent {
        timestamp: Utc::now(),
        action: "resolve".into(),
        item_ids: vec![item.id.clone()],
        actor: "agent".into(),
        note: args.attest.clone(),
        scores_before: Some(scores_before),
        scores_after: Some(scores_after.clone()),
    })?;

    println!(
        "Resolved {} -> {}",
        &item.id[..8.min(item.id.len())],
        updated.status.as_str()
    );
    println!("Summary:  {}", updated.summary);
    println!("Score:    {:.1}%", scores_after.overall * 100.0);

    Ok(())
}

fn cmd_log(args: &crate::args::CardloopLogArgs) -> anyhow::Result<()> {
    let log = ProgressionLog::open(&progression_path())?;
    let events = log.read_recent(args.count)?;

    if events.is_empty() {
        println!("No progression events yet. Run `cardloop scan` to start.");
        return Ok(());
    }

    for event in &events {
        let score_str = event
            .scores_after
            .as_ref()
            .map(|s| format!("{:.1}%", s.overall * 100.0))
            .unwrap_or_else(|| "-".into());

        println!(
            "[{}] {} ({} items) score={} {}",
            event.timestamp.format("%Y-%m-%d %H:%M"),
            event.action,
            event.item_ids.len(),
            score_str,
            event.note.as_deref().unwrap_or("")
        );
    }

    Ok(())
}
