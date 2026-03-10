mod support;

use std::time::Duration;

use anyhow::Result;
use axum::http::StatusCode;
use serde_json::json;

use support::TestStack;

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
            &json!({ "force_reindex": false, "run_at": null }),
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
                "force_reindex": false,
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
                "force_reindex": false,
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

    let (status_code, _) = stack
        .post_json(
            "/search",
            &json!({ "query": "Ownership" }),
            &[("x-api-key", "secret-key")],
        )
        .await?;
    assert_eq!(status_code, StatusCode::OK);

    Ok(())
}
