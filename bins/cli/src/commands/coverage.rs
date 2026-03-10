use crate::args::CoverageArgs;

/// Show topic coverage metrics.
pub async fn run(args: &CoverageArgs) -> anyhow::Result<()> {
    anyhow::bail!(
        "coverage CLI is not wired to the analytics service yet for topic {}",
        args.topic
    );
}
