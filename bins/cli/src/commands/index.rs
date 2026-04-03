use common::ReindexMode;
use surface_runtime::SurfaceServices;

use crate::args::IndexArgs;
use crate::output;
use crate::usecases::{self, IndexRequest, RuntimeHandles};

pub async fn run(args: &IndexArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let reindex_mode = if args.force {
        ReindexMode::Force
    } else {
        ReindexMode::Incremental
    };
    let summary = usecases::index(handles, IndexRequest { reindex_mode }, None).await?;
    output::print_index_summary(&summary);
    Ok(())
}
