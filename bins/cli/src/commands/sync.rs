use std::path::Path;

use crate::args::SyncArgs;
use crate::output::ensure_path_exists;

/// Sync an Anki collection to the index.
pub async fn run(args: &SyncArgs) -> anyhow::Result<()> {
    let source = Path::new(&args.source);
    ensure_path_exists(source, "source file")?;
    anyhow::bail!(
        "sync CLI is not wired to the sync service yet for source {}",
        args.source
    );
}
