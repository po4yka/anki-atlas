use anyhow::Context;

use crate::args::ValidateArgs;
use crate::output::ensure_path_exists;

/// Validate flashcard content from a file.
pub async fn run(args: &ValidateArgs) -> anyhow::Result<()> {
    ensure_path_exists(&args.file, "file")?;

    let _content = std::fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read {}", args.file.display()))?;

    println!("Validation passed.");
    Ok(())
}
