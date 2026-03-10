use surface_runtime::SurfaceServices;

use crate::args::GapsArgs;
use crate::output;

pub async fn run(args: &GapsArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let gaps = services
        .analytics
        .get_gaps(args.topic.clone(), args.min_coverage)
        .await?;
    output::print_gaps(&gaps);
    Ok(())
}
