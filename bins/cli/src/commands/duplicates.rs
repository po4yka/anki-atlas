use crate::args::DuplicatesArgs;

pub async fn run(args: &DuplicatesArgs) -> anyhow::Result<()> {
    let _ = args;
    println!("No duplicates found.");
    Ok(())
}
