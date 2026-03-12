pub async fn run(settings: &common::config::Settings) -> anyhow::Result<()> {
    let result = crate::usecases::run_migrations(settings).await?;
    println!("applied: {}", result.applied.join(", "));
    println!("skipped: {}", result.skipped.join(", "));
    Ok(())
}
