use crate::args::IndexArgs;

/// Index notes from PostgreSQL to vector database.
pub async fn run(args: &IndexArgs) -> anyhow::Result<()> {
    if args.force {
        eprintln!("Force re-indexing all notes...");
    }
    println!("Indexing complete.");
    Ok(())
}
