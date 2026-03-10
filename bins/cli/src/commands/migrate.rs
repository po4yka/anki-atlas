pub async fn run(settings: &common::config::Settings) -> anyhow::Result<()> {
    let pool = database::create_pool(&settings.database()).await?;
    let result = database::run_migrations(&pool).await?;
    println!("applied: {}", result.applied.join(", "));
    println!("skipped: {}", result.skipped.join(", "));
    Ok(())
}
