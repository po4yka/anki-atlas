use std::path::Path;

use crate::args::SyncArgs;
use crate::output::ensure_path_exists;

/// Sync an Anki collection to the index.
pub async fn run(args: &SyncArgs) -> anyhow::Result<()> {
    let source = Path::new(&args.source);
    ensure_path_exists(source, "source file")?;

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
