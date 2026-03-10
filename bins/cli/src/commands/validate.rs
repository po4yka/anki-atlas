use crate::args::ValidateArgs;
use crate::output;

pub async fn run(args: &ValidateArgs) -> anyhow::Result<()> {
    let summary =
        surface_runtime::ValidationService::new().validate_file(&args.file, args.quality)?;
    output::print_validation(&summary);
    Ok(())
}
