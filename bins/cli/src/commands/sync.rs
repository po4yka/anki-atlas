use surface_runtime::SurfaceServices;

use crate::args::SyncArgs;
use crate::output;

pub async fn run(args: &SyncArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let summary = services
        .sync
        .sync_collection(
            args.source.clone(),
            !args.no_migrate,
            !args.no_index,
            args.force_reindex,
        )
        .await?;
    output::print_sync_summary(&summary);
    Ok(())
}
