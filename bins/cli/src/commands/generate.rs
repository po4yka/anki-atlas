use crate::args::GenerateArgs;
use crate::output;
use crate::usecases::{self, GenerateRequest};

pub async fn run(args: &GenerateArgs) -> anyhow::Result<()> {
    let preview = usecases::generate_preview(
        &surface_runtime::GeneratePreviewService::new(),
        &GenerateRequest {
            file: args.file.clone(),
        },
    )?;
    output::print_generate_preview(&preview);
    if !args.dry_run {
        println!("preview-only workflow: no cards were persisted");
    }
    Ok(())
}
