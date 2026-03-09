use crate::args::ObsidianSyncArgs;
use crate::output::ensure_path_exists;

/// Scan an Obsidian vault and preview or sync cards.
pub async fn run(args: &ObsidianSyncArgs) -> anyhow::Result<()> {
    ensure_path_exists(&args.vault, "vault")?;

    println!("Scanning vault: {}", args.vault.display());
    Ok(())
}
