use surface_runtime::SurfaceServices;

use crate::args::DuplicatesArgs;
use crate::output;
use crate::usecases::{self, DuplicatesRequest, RuntimeHandles};

pub async fn run(args: &DuplicatesArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let (clusters, stats) = usecases::duplicates(
        handles,
        DuplicatesRequest {
            threshold: args.threshold,
            max: args.max,
            deck_names: args.deck_names.clone(),
            tags: args.tags.clone(),
        },
    )
    .await?;
    output::print_duplicates(&clusters, &stats, args.verbose);
    Ok(())
}
