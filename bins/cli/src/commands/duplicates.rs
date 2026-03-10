use crate::args::DuplicatesArgs;

/// Find near-duplicate notes.
pub async fn run(_args: &DuplicatesArgs) -> anyhow::Result<()> {
    anyhow::bail!("duplicates CLI is not wired to the analytics service yet");
}
