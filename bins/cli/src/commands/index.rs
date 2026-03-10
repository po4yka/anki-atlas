use surface_runtime::SurfaceServices;

use crate::args::IndexArgs;
use crate::output;

pub async fn run(args: &IndexArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let summary = services.index.index_all_notes(args.force).await?;
    output::print_index_summary(&summary);
    Ok(())
}
