use surface_runtime::SurfaceServices;

use crate::args::WeakNotesArgs;
use crate::output;

pub async fn run(args: &WeakNotesArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let notes = services
        .analytics
        .get_weak_notes(args.topic.clone(), args.limit)
        .await?;
    output::print_weak_notes(&notes);
    Ok(())
}
