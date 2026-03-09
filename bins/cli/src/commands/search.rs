use crate::args::SearchArgs;

pub async fn run(args: &SearchArgs) -> anyhow::Result<()> {
    println!("Searching for: {}", args.query);
    println!("No results found.");
    Ok(())
}
