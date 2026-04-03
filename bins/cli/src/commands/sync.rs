use common::ReindexMode;
use surface_runtime::SurfaceServices;

use crate::args::SyncArgs;
use crate::output;
use crate::usecases::{self, RuntimeHandles, SyncRequest};

pub async fn run(args: &SyncArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let reindex_mode = if args.force_reindex {
        ReindexMode::Force
    } else {
        ReindexMode::Incremental
    };
    let summary = usecases::sync(
        handles,
        SyncRequest {
            source: args.source.clone(),
            run_migrations: !args.no_migrate,
            run_index: !args.no_index,
            reindex_mode,
        },
        None,
    )
    .await?;
    output::print_sync_summary(&summary);
    Ok(())
}
