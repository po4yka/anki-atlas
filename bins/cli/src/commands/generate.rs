use anyhow::Context;

use crate::args::GenerateArgs;

pub async fn run(args: &GenerateArgs) -> anyhow::Result<()> {
    if !args.file.exists() {
        anyhow::bail!("file not found: {}", args.file.display());
    }

    let _content = std::fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read {}", args.file.display()))?;

    println!("Generate from: {}", args.file.display());
    Ok(())
}
