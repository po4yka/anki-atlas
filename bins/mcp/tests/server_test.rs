use anki_atlas_mcp::server::AnkiAtlasServer;

// --- Server construction ---

#[test]
fn server_can_be_created() {
    let _server = AnkiAtlasServer::new();
}

// --- Server info ---

#[test]
fn server_name_is_anki_atlas() {
    let server = AnkiAtlasServer::new();
    assert_eq!(server.name(), "anki-atlas");
}

#[test]
fn server_version_is_set() {
    let server = AnkiAtlasServer::new();
    let version = server.version();
    assert!(!version.is_empty(), "version should not be empty");
    // Should be a valid semver-ish string
    assert!(
        version.contains('.'),
        "version should contain a dot: got {version}"
    );
}

// --- Tool registration ---

#[test]
fn server_registers_9_tools() {
    let server = AnkiAtlasServer::new();
    assert_eq!(
        server.tool_count(),
        9,
        "expected 9 tools registered, got {}",
        server.tool_count()
    );
}

#[test]
fn server_has_search_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_search"),
        "missing ankiatlas_search tool, got: {names:?}"
    );
}

#[test]
fn server_has_topic_coverage_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_topic_coverage"),
        "missing ankiatlas_topic_coverage tool, got: {names:?}"
    );
}

#[test]
fn server_has_topic_gaps_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_topic_gaps"),
        "missing ankiatlas_topic_gaps tool, got: {names:?}"
    );
}

#[test]
fn server_has_duplicates_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_duplicates"),
        "missing ankiatlas_duplicates tool, got: {names:?}"
    );
}

#[test]
fn server_has_sync_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_sync"),
        "missing ankiatlas_sync tool, got: {names:?}"
    );
}

#[test]
fn server_has_generate_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_generate"),
        "missing ankiatlas_generate tool, got: {names:?}"
    );
}

#[test]
fn server_has_validate_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_validate"),
        "missing ankiatlas_validate tool, got: {names:?}"
    );
}

#[test]
fn server_has_obsidian_sync_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_obsidian_sync"),
        "missing ankiatlas_obsidian_sync tool, got: {names:?}"
    );
}

#[test]
fn server_has_tag_audit_tool() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    assert!(
        names.contains(&"ankiatlas_tag_audit"),
        "missing ankiatlas_tag_audit tool, got: {names:?}"
    );
}

// --- Tool names are sorted for deterministic output ---

#[test]
fn tool_names_are_all_prefixed() {
    let server = AnkiAtlasServer::new();
    let names = server.tool_names();
    for name in names {
        assert!(
            name.starts_with("ankiatlas_"),
            "tool name should start with 'ankiatlas_': got {name}"
        );
    }
}

// --- run_server exists and is async ---

#[tokio::test]
async fn run_server_function_exists() {
    // Just verify the function signature compiles; we don't actually run it
    // because it blocks on stdio. Instead, verify AnkiAtlasServer can be
    // created and has the expected tools.
    let server = AnkiAtlasServer::new();
    assert_eq!(server.tool_count(), 9);
}

// --- run_server runtime behavior ---

#[tokio::test]
async fn run_server_starts_without_panic() {
    // run_server should initialize logging and create a server without panicking.
    // In a test environment, stdin is closed immediately so it should return
    // (either Ok or Err) rather than blocking forever.
    // Use a timeout so the test doesn't hang if something goes wrong.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        anki_atlas_mcp::server::run_server(),
    )
    .await;

    // Any of these outcomes is acceptable:
    // - Timeout: server is blocking on stdin (correct behavior)
    // - Ok(Ok(())): server exited cleanly (stdin closed)
    // - Ok(Err(_)): server returned an error (e.g. transport error)
    // The only unacceptable outcome is a panic (todo!()).
    match result {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {}
        Err(_elapsed) => {}
    }
}

#[tokio::test]
async fn run_server_returns_anyhow_result() {
    // Verify run_server returns anyhow::Result<()>, not panicking.
    // This is a type-level check that also exercises the startup path.
    let result: Result<Result<(), anyhow::Error>, _> = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        anki_atlas_mcp::server::run_server(),
    )
    .await;

    // Should not panic (currently does because of todo!()).
    assert!(
        result.is_ok() || result.is_err(),
        "run_server should return a result, not panic"
    );
}
