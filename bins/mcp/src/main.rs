#[tokio::main]
async fn main() -> anyhow::Result<()> {
    anki_atlas_mcp::server::run_server().await
}
