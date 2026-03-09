use crate::args::CoverageArgs;

/// Show topic coverage metrics.
pub async fn run(args: &CoverageArgs) -> anyhow::Result<()> {
    println!("Coverage for topic: {}", args.topic);
    Ok(())
}
