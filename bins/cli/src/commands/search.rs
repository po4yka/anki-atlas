use crate::args::SearchArgs;

/// Search the Anki index.
pub async fn run(args: &SearchArgs) -> anyhow::Result<()> {
    anyhow::bail!(
        "search CLI is not wired to the search service yet for query {}",
        args.query
    );
}
