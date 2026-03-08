use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObsidianError {
    #[error("parse error: {message} (path: {path:?})")]
    Parse {
        message: String,
        path: Option<PathBuf>,
    },

    #[error("file not found: {0}")]
    NotFound(PathBuf),

    #[error("file too large: {path} ({size} bytes, max {max} bytes)")]
    FileTooLarge { path: PathBuf, size: u64, max: u64 },

    #[error("path outside vault root: {path} (root: {root})")]
    OutsideVault { path: PathBuf, root: PathBuf },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    Yaml(String),
}
