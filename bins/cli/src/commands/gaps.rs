use crate::args::GapsArgs;

pub async fn run(args: &GapsArgs) -> anyhow::Result<()> {
    println!("Gaps for topic: {}", args.topic);
    Ok(())
}
