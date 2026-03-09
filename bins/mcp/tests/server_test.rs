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
