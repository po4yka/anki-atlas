use surface_runtime::SurfaceServices;

use crate::args::CoverageArgs;
use crate::output;
use crate::usecases::{self, CoverageRequest, RuntimeHandles};

pub async fn run(args: &CoverageArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let coverage = usecases::coverage(
        handles,
        CoverageRequest {
            topic: args.topic.clone(),
            include_subtree: !args.no_subtree,
        },
    )
    .await?;
    output::print_coverage(&coverage);
    Ok(())
}
