use crate::args::TopicsArgs;

/// Manage topic taxonomy.
pub async fn run(args: &TopicsArgs) -> anyhow::Result<()> {
    if let Some(ref file) = args.file {
        println!("Loading topics from {}", file.display());
    } else {
        println!("Listing topics.");
    }
    Ok(())
}
