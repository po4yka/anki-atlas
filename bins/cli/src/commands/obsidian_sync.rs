use crate::args::ObsidianSyncArgs;
use crate::output;

pub async fn run(args: &ObsidianSyncArgs) -> anyhow::Result<()> {
    let preview = surface_runtime::ObsidianScanService::new().scan(
        &args.vault,
        &args.source_dirs,
        args.dry_run,
    )?;
    output::print_obsidian_scan(&preview);
    Ok(())
}
