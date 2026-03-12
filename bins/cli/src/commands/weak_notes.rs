use surface_runtime::SurfaceServices;

use crate::args::WeakNotesArgs;
use crate::output;
use crate::usecases::{self, RuntimeHandles, WeakNotesRequest};

pub async fn run(args: &WeakNotesArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let notes = usecases::weak_notes(
        handles,
        WeakNotesRequest {
            topic: args.topic.clone(),
            limit: args.limit,
        },
    )
    .await?;
    output::print_weak_notes(&notes);
    Ok(())
}
