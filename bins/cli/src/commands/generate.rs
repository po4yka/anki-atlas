use anyhow::Context;

use crate::args::GenerateArgs;
use crate::output::ensure_path_exists;

/// Parse an Obsidian note and preview card generation.
pub async fn run(args: &GenerateArgs) -> anyhow::Result<()> {
    ensure_path_exists(&args.file, "file")?;

    let note_markdown = std::fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read {}", args.file.display()))?;

    println!(
        "Generate from: {} ({} bytes)",
        args.file.display(),
        note_markdown.len()
    );
    Ok(())
}
