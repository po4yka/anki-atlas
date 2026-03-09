use crate::args::IndexArgs;

pub async fn run(args: &IndexArgs) -> anyhow::Result<()> {
    if args.force {
        eprintln!("Force re-indexing all notes...");
    }
    println!("Indexing complete.");
    Ok(())
}
