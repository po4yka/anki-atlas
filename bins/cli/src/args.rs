use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Top-level CLI arguments.
#[derive(Parser, Debug)]
#[command(
    name = "anki-atlas",
    about = "Searchable hybrid index for Anki collections"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Show version information.
    Version,

    /// Run database migrations.
    Migrate,

    /// Parse an Obsidian note and preview card generation.
    Generate(GenerateArgs),

    /// Validate flashcard content from a file.
    Validate(ValidateArgs),

    /// Scan an Obsidian vault and preview or sync cards.
    ObsidianSync(ObsidianSyncArgs),

    /// Audit tags for convention violations.
    TagAudit(TagAuditArgs),
}

/// Arguments for the `generate` command.
#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Path to an Obsidian markdown note.
    pub file: PathBuf,

    /// Preview without generating.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

/// Arguments for the `validate` command.
#[derive(clap::Args, Debug)]
pub struct ValidateArgs {
    /// File with card front/back (--- separated).
    pub file: PathBuf,

    /// Run quality assessment.
    #[arg(short, long, default_value_t = false)]
    pub quality: bool,
}

/// Arguments for the `obsidian-sync` command.
#[derive(clap::Args, Debug)]
pub struct ObsidianSyncArgs {
    /// Path to Obsidian vault.
    pub vault: PathBuf,

    /// Comma-separated subdirectories to scan.
    #[arg(short, long)]
    pub source_dirs: Option<String>,

    /// Scan only, do not generate/sync.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

/// Arguments for the `tag-audit` command.
#[derive(clap::Args, Debug)]
pub struct TagAuditArgs {
    /// File with tags, one per line.
    pub file: PathBuf,

    /// Show normalized tags.
    #[arg(short, long, default_value_t = false)]
    pub fix: bool,
}
