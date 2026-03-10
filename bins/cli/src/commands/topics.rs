use crate::args::TopicsArgs;

/// Manage topic taxonomy.
pub async fn run(args: &TopicsArgs) -> anyhow::Result<()> {
    if let Some(ref file) = args.file {
        anyhow::bail!(
            "topics CLI is not wired to taxonomy services yet for file {}",
            file.display()
        );
    }

    anyhow::bail!("topics CLI is not wired to taxonomy services yet");
}
