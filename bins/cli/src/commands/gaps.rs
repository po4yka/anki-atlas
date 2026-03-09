use crate::args::GapsArgs;

/// Detect gaps in topic coverage.
pub async fn run(args: &GapsArgs) -> anyhow::Result<()> {
    println!("Gaps for topic: {}", args.topic);
    Ok(())
}
