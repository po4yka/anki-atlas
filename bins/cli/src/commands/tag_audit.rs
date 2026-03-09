use crate::args::TagAuditArgs;

pub async fn run(args: &TagAuditArgs) -> anyhow::Result<()> {
    if !args.file.exists() {
        anyhow::bail!("file not found: {}", args.file.display());
    }

    println!("Tag audit: {}", args.file.display());
    Ok(())
}
