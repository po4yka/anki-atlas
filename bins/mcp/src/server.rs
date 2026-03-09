// MCP server setup, tool registration.

/// The MCP server struct that registers all anki-atlas tools.
#[derive(Default)]
pub struct AnkiAtlasServer;

impl AnkiAtlasServer {
    /// Create a new server instance with all tools registered.
    pub fn new() -> Self {
        todo!()
    }

    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        todo!()
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        todo!()
    }

    /// Return the server name.
    pub fn name(&self) -> &str {
        todo!()
    }

    /// Return the server version.
    pub fn version(&self) -> &str {
        todo!()
    }
}

/// Configure and run the MCP server over stdio transport.
///
/// Registers all tools with the rmcp server, configures logging
/// to stderr (to avoid polluting stdio MCP protocol), and runs
/// until stdin closes or SIGTERM is received.
pub async fn run_server() -> anyhow::Result<()> {
    todo!()
}
