use crate::args::ObsidianSyncArgs;

pub async fn run(args: &ObsidianSyncArgs) -> anyhow::Result<()> {
    if !args.vault.exists() {
        anyhow::bail!("vault not found: {}", args.vault.display());
    }

    println!("Scanning vault: {}", args.vault.display());
    Ok(())
}
