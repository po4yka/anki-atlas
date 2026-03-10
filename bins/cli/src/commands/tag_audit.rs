use crate::args::TagAuditArgs;
use crate::output;

pub async fn run(args: &TagAuditArgs) -> anyhow::Result<()> {
    let summary = surface_runtime::TagAuditService::new().audit_file(&args.file, args.fix)?;
    output::print_tag_audit(&summary);
    Ok(())
}
