use crate::args::GapsArgs;

/// Detect gaps in topic coverage.
pub async fn run(args: &GapsArgs) -> anyhow::Result<()> {
    anyhow::bail!(
        "gaps CLI is not wired to the analytics service yet for topic {}",
        args.topic
    );
}
