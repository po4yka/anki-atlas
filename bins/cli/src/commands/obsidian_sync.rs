use crate::args::ObsidianSyncArgs;
use crate::output;
use crate::usecases::{self, ObsidianScanRequest};

pub async fn run(args: &ObsidianSyncArgs) -> anyhow::Result<()> {
    let preview = usecases::obsidian_scan(
        &surface_runtime::ObsidianScanService::new(),
        &ObsidianScanRequest {
            vault: args.vault.clone(),
            source_dirs: args.source_dirs.clone(),
            execution_mode: if args.dry_run {
                common::ExecutionMode::DryRun
            } else {
                common::ExecutionMode::Execute
            },
        },
        None,
    )?;
    output::print_obsidian_scan(&preview);
    Ok(())
}
