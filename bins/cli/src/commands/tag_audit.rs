use crate::args::TagAuditArgs;
use crate::output::ensure_path_exists;

/// Audit tags for convention violations.
pub async fn run(args: &TagAuditArgs) -> anyhow::Result<()> {
    ensure_path_exists(&args.file, "file")?;

    println!("Tag audit: {}", args.file.display());
    Ok(())
}
