pub fn run() -> anyhow::Result<()> {
    println!("{}", env!("CARGO_PKG_VERSION"));
    Ok(())
}
