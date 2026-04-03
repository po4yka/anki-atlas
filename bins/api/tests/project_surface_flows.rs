mod support;

use std::time::Duration;

use anyhow::Result;
use axum::http::StatusCode;
use serde_json::json;

use support::{SeedNote, TestStack};

async fn maybe_stack(api_key: Option<&str>) -> Result<Option<TestStack>> {
    match TestStack::new(api_key).await {
        Ok(stack) => Ok(Some(stack)),
        Err(error)
            if format!("{error:#}").contains("/var/run/docker.sock")
                || format!("{error:#}").contains("failed to initialize a docker client") =>
        {
            eprintln!("skipping docker-backed surface integration test: {error}");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[tokio::test]
async fn cli_sync_and_index_then_api_search() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("cli-sync.anki2")?;

    let output = stack.run_cli(&["sync", fixture.to_str().unwrap()])?;
    assert!(
        output.status.success(),
        "cli sync failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(stack.notes_count().await?, 1);
    assert_eq!(stack.cards_count().await?, 1);
    assert!(stack.qdrant_point_count().await? > 0);

    let search = stack.api_search("Ownership").await?;
    let note_ids = search["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["note_id"].as_i64())
        .collect::<Vec<_>>();
    assert!(
        note_ids.contains(&100),
        "expected synced note 100 in search response: {search}"
    );

    Ok(())
}

#[tokio::test]
async fn api_enqueue_index_job_then_worker_completes() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("index-job.anki2")?;

    let output = stack.run_cli(&["sync", fixture.to_str().unwrap(), "--no-index"])?;
    assert!(
        output.status.success(),
        "cli sync without index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(stack.qdrant_point_count().await?, 0);

    let (status_code, accepted_body) = stack
        .post_json(
            "/jobs/index",
            &json!({ "reindex_mode": "incremental", "run_at": null }),
            &[],
        )
        .await?;
    assert_eq!(status_code, StatusCode::ACCEPTED);
    let job_id = accepted_body["job_id"].as_str().unwrap().to_string();

    let mut worker = stack.spawn_worker()?;
    let terminal = stack
        .poll_job_terminal(&job_id, Duration::from_secs(30))
        .await?;
    assert_eq!(terminal["status"], "succeeded");
    worker.stop()?;

    assert!(stack.qdrant_point_count().await? > 0);
    let search = stack.api_search("Ownership").await?;
    assert!(
        search["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["note_id"] == 100),
        "expected note 100 after index job: {search}"
    );

    Ok(())
}

#[tokio::test]
async fn api_enqueue_sync_job_then_worker_updates_job_status() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("sync-job.anki2")?;

    let (status_code, accepted_body) = stack
        .post_json(
            "/jobs/sync",
            &json!({
                "source": fixture,
                "run_migrations": true,
                "index": true,
                "reindex_mode": "incremental",
                "run_at": null
            }),
            &[],
        )
        .await?;
    assert_eq!(status_code, StatusCode::ACCEPTED);
    let job_id = accepted_body["job_id"].as_str().unwrap().to_string();

    let mut worker = stack.spawn_worker()?;
    let terminal = stack
        .poll_job_terminal(&job_id, Duration::from_secs(30))
        .await?;
    assert_eq!(terminal["status"], "succeeded");
    worker.stop()?;

    assert_eq!(stack.notes_count().await?, 1);
    assert_eq!(stack.cards_count().await?, 1);
    assert!(stack.qdrant_point_count().await? > 0);

    Ok(())
}

#[tokio::test]
async fn api_cancel_job_before_execution() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("cancel-job.anki2")?;

    let (status_code, accepted_body) = stack
        .post_json(
            "/jobs/sync",
            &json!({
                "source": fixture,
                "run_migrations": true,
                "index": true,
                "reindex_mode": "incremental",
                "run_at": null
            }),
            &[],
        )
        .await?;
    assert_eq!(status_code, StatusCode::ACCEPTED);
    let job_id = accepted_body["job_id"].as_str().unwrap().to_string();

    let (status_code, cancelled_body) = stack
        .post_json(&format!("/jobs/{job_id}/cancel"), &json!({}), &[])
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(cancelled_body["status"], "cancel_requested");
    assert_eq!(cancelled_body["cancel_requested"], true);

    let mut worker = stack.spawn_worker()?;
    let terminal = stack
        .poll_job_terminal(&job_id, Duration::from_secs(30))
        .await?;
    assert_eq!(terminal["status"], "cancelled");
    worker.stop()?;

    assert_eq!(stack.notes_count().await?, 0);
    assert_eq!(stack.qdrant_point_count().await?, 0);

    Ok(())
}

#[tokio::test]
async fn api_key_auth_works_with_real_services() -> Result<()> {
    let Some(stack) = maybe_stack(Some("secret-key")).await? else {
        return Ok(());
    };

    let (status_code, _) = stack.get_json("/health", &[]).await?;
    assert_eq!(status_code, StatusCode::OK);

    let (status_code, _) = stack.get_json("/topics", &[]).await?;
    assert_eq!(status_code, StatusCode::UNAUTHORIZED);

    let (status_code, topics) = stack
        .get_json("/topics", &[("x-api-key", "secret-key")])
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(topics["topics"].as_array().unwrap().len(), 0);

    let (status_code, _) = stack
        .post_json("/search", &json!({ "query": "Ownership" }), &[])
        .await?;
    assert_eq!(status_code, StatusCode::UNAUTHORIZED);

    let (status_code, _) = stack
        .post_json(
            "/search",
            &json!({ "query": "Ownership" }),
            &[("x-api-key", "wrong-key")],
        )
        .await?;
    assert_eq!(status_code, StatusCode::UNAUTHORIZED);

    Ok(())
}

#[tokio::test]
async fn cli_topics_load_then_api_topics_tree_and_ready() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let taxonomy = stack.create_taxonomy_fixture("topics-tree.yaml")?;

    let output = stack.run_cli(&[
        "topics",
        "load",
        "--file",
        taxonomy.to_str().expect("taxonomy fixture path"),
    ])?;
    assert!(
        output.status.success(),
        "topics load failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("taxonomy loaded"),
        "unexpected stdout: {stdout}"
    );
    assert!(stdout.contains("topics: 3"), "unexpected stdout: {stdout}");
    assert!(stdout.contains("roots: 1"), "unexpected stdout: {stdout}");

    let (status_code, ready) = stack.get_json("/ready", &[]).await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(ready["status"], "ready");

    let (status_code, topics) = stack.get_json("/topics?root_path=rust", &[]).await?;
    assert_eq!(status_code, StatusCode::OK);
    let root = &topics["topics"][0];
    assert_eq!(root["path"], "rust");
    assert_eq!(root["children"].as_array().unwrap().len(), 2);
    assert_eq!(root["children"][0]["path"], "rust/borrowing");
    assert_eq!(root["children"][1]["path"], "rust/ownership");

    Ok(())
}

#[tokio::test]
async fn api_topic_coverage_and_cli_reports_seeded_assignments() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let taxonomy = stack.create_taxonomy_fixture("coverage-taxonomy.yaml")?;

    let load = stack.run_cli(&[
        "topics",
        "load",
        "--file",
        taxonomy.to_str().expect("taxonomy fixture path"),
    ])?;
    assert!(
        load.status.success(),
        "topics load failed: {}",
        String::from_utf8_lossy(&load.stderr)
    );

    stack
        .seed_note(SeedNote {
            note_id: 700,
            card_id: 1700,
            deck_id: 11,
            deck_name: "Rust",
            model_id: 1234567890,
            model_name: "Basic",
            tags: vec!["ownership", "integration"],
            normalized_text: "Rust ownership move semantics and borrowing rules",
            raw_fields: "Front\x1fBack",
            ivl: 30,
            lapses: 4,
            reps: 18,
            fail_rate: Some(0.22),
        })
        .await?;
    stack
        .assign_topic(700, "rust", 0.72, "integration_seed")
        .await?;
    stack
        .assign_topic(700, "rust/ownership", 0.91, "integration_seed")
        .await?;

    let (status_code, coverage) = stack
        .get_json("/topic-coverage?topic_path=rust&include_subtree=true", &[])
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(coverage["path"], "rust");
    assert_eq!(coverage["note_count"], 1);
    assert_eq!(coverage["child_count"], 2);
    assert_eq!(coverage["covered_children"], 1);
    assert_eq!(coverage["mature_count"], 1);
    assert_eq!(coverage["weak_notes"], 1);
    assert!(
        coverage["avg_confidence"].as_f64().unwrap_or_default() > 0.7,
        "unexpected coverage payload: {coverage}"
    );

    let output = stack.run_cli(&["coverage", "rust"])?;
    assert!(
        output.status.success(),
        "coverage cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("topic: rust (Rust)"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("note_count: 1"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("covered_children: 1/2"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("weak_notes: 1"),
        "unexpected stdout: {stdout}"
    );

    Ok(())
}

#[tokio::test]
async fn api_topic_gaps_and_weak_notes_reflect_seeded_taxonomy() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };
    let taxonomy = stack.create_taxonomy_fixture("gaps-taxonomy.yaml")?;

    let load = stack.run_cli(&[
        "topics",
        "load",
        "--file",
        taxonomy.to_str().expect("taxonomy fixture path"),
    ])?;
    assert!(
        load.status.success(),
        "topics load failed: {}",
        String::from_utf8_lossy(&load.stderr)
    );

    stack
        .seed_note(SeedNote {
            note_id: 701,
            card_id: 1701,
            deck_id: 11,
            deck_name: "Rust",
            model_id: 1234567890,
            model_name: "Basic",
            tags: vec!["weak", "ownership"],
            normalized_text: "Ownership errors and borrow checker fixes",
            raw_fields: "Front\x1fBack",
            ivl: 25,
            lapses: 6,
            reps: 20,
            fail_rate: Some(0.35),
        })
        .await?;
    stack
        .assign_topic(701, "rust/ownership", 0.94, "integration_seed")
        .await?;

    let (status_code, gaps) = stack
        .get_json("/topic-gaps?topic_path=rust&min_coverage=1", &[])
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(gaps["root_path"], "rust");
    assert_eq!(gaps["missing_count"], 2);
    assert_eq!(gaps["undercovered_count"], 0);
    let gap_paths = gaps["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|gap| gap["path"].as_str())
        .collect::<Vec<_>>();
    assert!(
        gap_paths.contains(&"rust"),
        "expected root gap in response: {gaps}"
    );
    assert!(
        gap_paths.contains(&"rust/borrowing"),
        "expected borrowing gap in response: {gaps}"
    );

    let (status_code, weak_notes) = stack
        .get_json(
            "/topic-weak-notes?topic_path=rust/ownership&max_results=5",
            &[],
        )
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(weak_notes["notes"][0]["note_id"], 701);
    assert_eq!(weak_notes["notes"][0]["topic_path"], "rust/ownership");
    assert!(
        weak_notes["notes"][0]["fail_rate"]
            .as_f64()
            .unwrap_or_default()
            >= 0.3,
        "unexpected weak notes payload: {weak_notes}"
    );

    let gaps_output = stack.run_cli(&["gaps", "rust", "--min-coverage", "1"])?;
    assert!(
        gaps_output.status.success(),
        "gaps cli failed: {}",
        String::from_utf8_lossy(&gaps_output.stderr)
    );
    let gaps_stdout = String::from_utf8_lossy(&gaps_output.stdout);
    assert!(
        gaps_stdout.contains("gaps: 2"),
        "unexpected stdout: {gaps_stdout}"
    );
    assert!(
        gaps_stdout.contains("- rust/borrowing [\"missing\"] notes=0 threshold=1"),
        "unexpected stdout: {gaps_stdout}"
    );

    let weak_output = stack.run_cli(&["weak-notes", "rust/ownership", "--limit", "5"])?;
    assert!(
        weak_output.status.success(),
        "weak-notes cli failed: {}",
        String::from_utf8_lossy(&weak_output.stderr)
    );
    let weak_stdout = String::from_utf8_lossy(&weak_output.stdout);
    assert!(
        weak_stdout.contains("weak_notes: 1"),
        "unexpected stdout: {weak_stdout}"
    );
    assert!(
        weak_stdout.contains("- note=701 confidence=0.940 lapses=6"),
        "unexpected stdout: {weak_stdout}"
    );

    Ok(())
}

#[tokio::test]
async fn api_and_cli_duplicates_detect_indexed_notes() -> Result<()> {
    let Some(stack) = maybe_stack(None).await? else {
        return Ok(());
    };

    stack
        .seed_note(SeedNote {
            note_id: 800,
            card_id: 1800,
            deck_id: 12,
            deck_name: "Rust",
            model_id: 1234567890,
            model_name: "Basic",
            tags: vec!["duplicate", "ownership"],
            normalized_text: "Rust ownership duplicate note",
            raw_fields: "Front\x1fBack",
            ivl: 35,
            lapses: 1,
            reps: 24,
            fail_rate: Some(0.05),
        })
        .await?;
    stack
        .seed_note(SeedNote {
            note_id: 801,
            card_id: 1801,
            deck_id: 12,
            deck_name: "Rust",
            model_id: 1234567890,
            model_name: "Basic",
            tags: vec!["duplicate", "ownership"],
            normalized_text: "Rust ownership duplicate note",
            raw_fields: "Front\x1fBack",
            ivl: 28,
            lapses: 2,
            reps: 8,
            fail_rate: Some(0.07),
        })
        .await?;
    stack
        .seed_note(SeedNote {
            note_id: 802,
            card_id: 1802,
            deck_id: 12,
            deck_name: "Rust",
            model_id: 1234567890,
            model_name: "Basic",
            tags: vec!["unique", "borrowing"],
            normalized_text: "Borrowing and lifetimes are distinct concepts",
            raw_fields: "Front\x1fBack",
            ivl: 14,
            lapses: 0,
            reps: 3,
            fail_rate: Some(0.01),
        })
        .await?;

    let index = stack.run_cli(&["index", "--force"])?;
    assert!(
        index.status.success(),
        "index cli failed: {}",
        String::from_utf8_lossy(&index.stderr)
    );
    assert!(stack.qdrant_point_count().await? >= 3);

    let (status_code, duplicates) = stack
        .get_json(
            "/duplicates?threshold=0.99&max_clusters=5&deck_filter[]=Rust&tag_filter[]=duplicate",
            &[],
        )
        .await?;
    assert_eq!(status_code, StatusCode::OK);
    assert_eq!(duplicates["stats"]["clusters_found"], 1);
    assert_eq!(duplicates["stats"]["total_duplicates"], 1);
    assert_eq!(duplicates["clusters"][0]["representative_id"], 800);
    assert_eq!(duplicates["clusters"][0]["size"], 2);
    assert_eq!(duplicates["clusters"][0]["duplicates"][0]["note_id"], 801);

    let output = stack.run_cli(&[
        "duplicates",
        "--threshold",
        "0.99",
        "--max",
        "5",
        "--deck",
        "Rust",
        "--tag",
        "duplicate",
        "--verbose",
    ])?;
    assert!(
        output.status.success(),
        "duplicates cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("clusters: 1"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("total_duplicates: 1"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("- representative=800 size=2 decks=Rust tags=duplicate,ownership"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("duplicate note=801 similarity=1.0000 decks=Rust tags=duplicate,ownership"),
        "unexpected stdout: {stdout}"
    );

    Ok(())
}
