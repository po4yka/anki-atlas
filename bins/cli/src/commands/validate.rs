use crate::args::ValidateArgs;
use crate::output;
use crate::usecases::{self, ValidateRequest};

pub async fn run(args: &ValidateArgs) -> anyhow::Result<()> {
    let summary = usecases::validate(
        &surface_runtime::ValidationService::new(),
        &ValidateRequest {
            file: args.file.clone(),
            include_quality: args.quality,
        },
    )?;
    output::print_validation(&summary);
    Ok(())
}
