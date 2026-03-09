use crate::args::CoverageArgs;

pub async fn run(args: &CoverageArgs) -> anyhow::Result<()> {
    println!("Coverage for topic: {}", args.topic);
    Ok(())
}
