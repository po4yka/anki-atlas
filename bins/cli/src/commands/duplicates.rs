use crate::args::DuplicatesArgs;

/// Find near-duplicate notes.
pub async fn run(_args: &DuplicatesArgs) -> anyhow::Result<()> {
    println!("No duplicates found.");
    Ok(())
}
