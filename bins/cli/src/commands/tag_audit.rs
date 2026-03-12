use crate::args::TagAuditArgs;
use crate::output;
use crate::usecases::{self, TagAuditRequest};

pub async fn run(args: &TagAuditArgs) -> anyhow::Result<()> {
    let summary = usecases::tag_audit(
        &surface_runtime::TagAuditService::new(),
        &TagAuditRequest {
            file: args.file.clone(),
            apply_fixes: args.fix,
        },
    )?;
    output::print_tag_audit(&summary);
    Ok(())
}
