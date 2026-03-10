mod support;

use anyhow::Result;
use sqlx::Row;

use support::TestStack;

async fn maybe_stack() -> Result<Option<TestStack>> {
    match TestStack::new(None).await {
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
async fn cli_search_returns_indexed_note_with_filters() -> Result<()> {
    let Some(stack) = maybe_stack().await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("cli-search.anki2")?;

    let sync = stack.run_cli(&["sync", fixture.to_str().expect("fixture path")])?;
    assert!(
        sync.status.success(),
        "cli sync failed: {}",
        String::from_utf8_lossy(&sync.stderr)
    );

    let output = stack.run_cli(&[
        "search",
        "Ownership",
        "--deck",
        "Default",
        "--tag",
        "rust",
        "--verbose",
    ])?;
    assert!(
        output.status.success(),
        "search cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("query: Ownership"),
        "unexpected stdout: {stdout}"
    );
    assert!(stdout.contains("results: 1"), "unexpected stdout: {stdout}");
    assert!(stdout.contains("- note=100"), "unexpected stdout: {stdout}");
    assert!(
        stdout.contains("sources=semantic,fts") || stdout.contains("sources=fts,semantic"),
        "unexpected stdout: {stdout}"
    );

    Ok(())
}

#[tokio::test]
async fn cli_topics_label_creates_topic_assignments() -> Result<()> {
    let Some(stack) = maybe_stack().await? else {
        return Ok(());
    };
    let fixture = stack.create_anki_fixture("cli-label.anki2")?;
    let taxonomy = stack.create_taxonomy_fixture("cli-label-topics.yaml")?;

    let sync = stack.run_cli(&[
        "sync",
        fixture.to_str().expect("fixture path"),
        "--no-index",
    ])?;
    assert!(
        sync.status.success(),
        "cli sync failed: {}",
        String::from_utf8_lossy(&sync.stderr)
    );

    let output = stack.run_cli(&[
        "topics",
        "label",
        "--file",
        taxonomy.to_str().expect("taxonomy path"),
        "--min-confidence",
        "0.0",
    ])?;
    assert!(
        output.status.success(),
        "topics label failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("labeling complete"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("notes_processed: 1"),
        "unexpected stdout: {stdout}"
    );

    let row =
        sqlx::query("SELECT COUNT(*) AS count FROM note_topics WHERE note_id = $1 AND method = $2")
            .bind(100_i64)
            .bind("embedding")
            .fetch_one(&stack.pool)
            .await?;
    let assignments: i64 = row.try_get("count")?;
    assert!(assignments > 0, "expected labeled topics for note 100");

    Ok(())
}

#[tokio::test]
async fn cli_migrate_reports_current_schema_state() -> Result<()> {
    let Some(stack) = maybe_stack().await? else {
        return Ok(());
    };

    let output = stack.run_cli(&["migrate"])?;
    assert!(
        output.status.success(),
        "migrate cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("applied:"), "unexpected stdout: {stdout}");
    assert!(stdout.contains("skipped:"), "unexpected stdout: {stdout}");
    assert!(
        stdout.contains("001_initial_schema"),
        "unexpected stdout: {stdout}"
    );
    assert!(
        stdout.contains("002_pg_trgm_lexical_search"),
        "unexpected stdout: {stdout}"
    );

    Ok(())
}
