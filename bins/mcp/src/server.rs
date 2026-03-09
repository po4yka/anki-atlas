// MCP server setup, tool registration.

use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt};
use tracing_subscriber::EnvFilter;

/// All registered tool names, sorted alphabetically.
const TOOL_NAMES: [&str; 9] = [
    "ankiatlas_duplicates",
    "ankiatlas_generate",
    "ankiatlas_obsidian_sync",
    "ankiatlas_search",
    "ankiatlas_sync",
    "ankiatlas_tag_audit",
    "ankiatlas_topic_coverage",
    "ankiatlas_topic_gaps",
    "ankiatlas_validate",
];

/// The MCP server struct that registers all anki-atlas tools.
#[derive(Default)]
pub struct AnkiAtlasServer;

impl AnkiAtlasServer {
    /// Create a new server instance with all tools registered.
    pub fn new() -> Self {
        Self
    }

    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        TOOL_NAMES.len()
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> &[&'static str] {
        &TOOL_NAMES
    }

    /// Return the server name.
    pub fn name(&self) -> &str {
        "anki-atlas"
    }

    /// Return the server version.
    pub fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }
}

impl ServerHandler for AnkiAtlasServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new(self.name(), self.version());
        info
    }
}

/// Configure and run the MCP server over stdio transport.
///
/// Registers all tools with the rmcp server, configures logging
/// to stderr (to avoid polluting stdio MCP protocol), and runs
/// until stdin closes or SIGTERM is received.
pub async fn run_server() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .try_init()
        .ok();

    let transport = rmcp::transport::stdio();
    let server = AnkiAtlasServer::new().serve(transport).await?;
    server.waiting().await?;

    Ok(())
}
