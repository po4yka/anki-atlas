use std::path::Path;

/// Verify a path exists, returning a user-friendly error if not.
pub fn ensure_path_exists(path: &Path, label: &str) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("{label} not found: {}", path.display());
    }
    Ok(())
}
