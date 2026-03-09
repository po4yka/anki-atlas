use crate::args::SearchArgs;

/// Search the Anki index.
pub async fn run(args: &SearchArgs) -> anyhow::Result<()> {
    println!("Searching for: {}", args.query);
    println!("No results found.");
    Ok(())
}
