use std::path::Path;

use crate::args::SyncArgs;

pub async fn run(args: &SyncArgs) -> anyhow::Result<()> {
    let source = Path::new(&args.source);
    if !source.exists() {
        anyhow::bail!("source file not found: {}", args.source);
    }

    if !args.no_migrate {
        eprintln!("Running migrations...");
    }

    eprintln!("Syncing from {}...", args.source);

    if !args.no_index {
        eprintln!("Indexing notes...");
    }

    println!("Sync complete.");
    Ok(())
}
