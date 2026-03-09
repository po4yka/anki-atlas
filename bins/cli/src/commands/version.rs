pub fn run() -> anyhow::Result<()> {
    println!("anki-atlas {}", env!("CARGO_PKG_VERSION"));
    Ok(())
}
