use surface_runtime::SurfaceServices;

use crate::args::DuplicatesArgs;
use crate::output;

pub async fn run(args: &DuplicatesArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let (clusters, stats) = services
        .analytics
        .find_duplicates(
            args.threshold,
            args.max,
            (!args.deck_names.is_empty()).then(|| args.deck_names.clone()),
            (!args.tags.is_empty()).then(|| args.tags.clone()),
        )
        .await?;
    output::print_duplicates(&clusters, &stats, args.verbose);
    Ok(())
}
