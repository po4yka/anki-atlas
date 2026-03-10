use surface_runtime::SurfaceServices;

use crate::args::CoverageArgs;
use crate::output;

pub async fn run(args: &CoverageArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let coverage = services
        .analytics
        .get_coverage(args.topic.clone(), !args.no_subtree)
        .await?;
    let coverage = coverage.ok_or_else(|| anyhow::anyhow!("topic not found: {}", args.topic))?;
    output::print_coverage(&coverage);
    Ok(())
}
