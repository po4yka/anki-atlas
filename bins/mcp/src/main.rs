#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    anki_atlas_mcp::server::run_server().await
}
