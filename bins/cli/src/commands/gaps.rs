use surface_runtime::SurfaceServices;

use crate::args::GapsArgs;
use crate::output;
use crate::usecases::{self, GapsRequest, RuntimeHandles};

pub async fn run(args: &GapsArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let gaps = usecases::gaps(
        handles,
        GapsRequest {
            topic: args.topic.clone(),
            min_coverage: args.min_coverage,
        },
    )
    .await?;
    output::print_gaps(&gaps);
    Ok(())
}
