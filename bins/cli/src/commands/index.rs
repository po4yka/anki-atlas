use crate::args::IndexArgs;

/// Index notes from PostgreSQL to vector database.
pub async fn run(args: &IndexArgs) -> anyhow::Result<()> {
    anyhow::bail!(
        "index CLI is not wired to the indexer service yet (force={})",
        args.force
    );
}
