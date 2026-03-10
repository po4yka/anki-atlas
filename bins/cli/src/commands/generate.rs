use crate::args::GenerateArgs;
use crate::output;

pub async fn run(args: &GenerateArgs) -> anyhow::Result<()> {
    let preview = surface_runtime::GeneratePreviewService::new().preview(&args.file)?;
    output::print_generate_preview(&preview);
    if !args.dry_run {
        println!("preview-only workflow: no cards were persisted");
    }
    Ok(())
}
