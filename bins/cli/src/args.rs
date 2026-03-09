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

    /// Sync Anki collection to the index.
    Sync(SyncArgs),

    /// Run database migrations.
    Migrate,

    /// Index notes from PostgreSQL to vector database.
    Index(IndexArgs),

    /// Search the Anki index.
    Search(SearchArgs),

    /// Manage topic taxonomy.
    Topics(TopicsArgs),

    /// Show topic coverage metrics.
    Coverage(CoverageArgs),

    /// Detect gaps in topic coverage.
    Gaps(GapsArgs),

    /// Find near-duplicate notes.
    Duplicates(DuplicatesArgs),

    /// Parse an Obsidian note and preview card generation.
    Generate(GenerateArgs),

    /// Validate flashcard content from a file.
    Validate(ValidateArgs),

    /// Scan an Obsidian vault and preview or sync cards.
    ObsidianSync(ObsidianSyncArgs),

    /// Audit tags for convention violations.
    TagAudit(TagAuditArgs),
}

/// Arguments for the `sync` command.
#[derive(clap::Args, Debug)]
pub struct SyncArgs {
    /// Path to collection.anki2 file.
    #[arg(short, long)]
    pub source: String,

    /// Skip database migrations before sync.
    #[arg(long, default_value_t = false)]
    pub no_migrate: bool,

    /// Skip indexing notes to vector database after sync.
    #[arg(long, default_value_t = false)]
    pub no_index: bool,

    /// Force re-embedding all notes.
    #[arg(long, default_value_t = false)]
    pub force_reindex: bool,
}

/// Arguments for the `index` command.
#[derive(clap::Args, Debug)]
pub struct IndexArgs {
    /// Force re-embedding all notes.
    #[arg(short, long, default_value_t = false)]
    pub force: bool,
}

/// Arguments for the `search` command.
#[derive(clap::Args, Debug)]
pub struct SearchArgs {
    /// Search query.
    pub query: String,

    /// Filter by deck name.
    #[arg(short, long)]
    pub deck: Option<String>,

    /// Filter by tag.
    #[arg(short, long)]
    pub tag: Option<String>,

    /// Number of results.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub top: usize,

    /// Use only semantic search.
    #[arg(long, default_value_t = false)]
    pub semantic: bool,

    /// Use only full-text search.
    #[arg(long, default_value_t = false)]
    pub fts: bool,

    /// Show detailed scores.
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

/// Arguments for the `topics` command.
#[derive(clap::Args, Debug)]
pub struct TopicsArgs {
    /// Path to topics.yml file to load.
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Label notes with topics after loading.
    #[arg(short, long, default_value_t = false)]
    pub label: bool,

    /// Minimum confidence for labeling.
    #[arg(long, default_value_t = 0.3)]
    pub min_confidence: f64,
}

/// Arguments for the `coverage` command.
#[derive(clap::Args, Debug)]
pub struct CoverageArgs {
    /// Topic path (e.g., programming/python).
    pub topic: String,

    /// Exclude child topics.
    #[arg(long, default_value_t = false)]
    pub no_subtree: bool,
}

/// Arguments for the `gaps` command.
#[derive(clap::Args, Debug)]
pub struct GapsArgs {
    /// Topic path.
    pub topic: String,

    /// Minimum notes for coverage.
    #[arg(short, long, default_value_t = 1)]
    pub min_coverage: usize,
}

/// Arguments for the `duplicates` command.
#[derive(clap::Args, Debug)]
pub struct DuplicatesArgs {
    /// Similarity threshold (0-1).
    #[arg(short, long, default_value_t = 0.92)]
    pub threshold: f64,

    /// Maximum clusters to show.
    #[arg(short = 'n', long, default_value_t = 50)]
    pub max: usize,

    /// Filter by deck name.
    #[arg(short, long)]
    pub deck: Option<String>,

    /// Filter by tag.
    #[arg(long)]
    pub tag: Option<String>,

    /// Show all duplicates in clusters.
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
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
