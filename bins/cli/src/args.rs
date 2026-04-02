use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "anki-atlas",
    about = "Searchable hybrid index and analytics surface for Anki collections"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Version,
    Migrate,
    Tui,
    Sync(SyncArgs),
    Index(IndexArgs),
    Search(SearchArgs),
    Topics(TopicsArgs),
    Coverage(CoverageArgs),
    Gaps(GapsArgs),
    WeakNotes(WeakNotesArgs),
    Duplicates(DuplicatesArgs),
    Generate(GenerateArgs),
    Validate(ValidateArgs),
    ObsidianSync(ObsidianSyncArgs),
    TagAudit(TagAuditArgs),
    /// Card quality loop: scan, queue, fix, resolve
    Cardloop(CardloopArgs),
}

#[derive(Args, Debug)]
pub struct SyncArgs {
    pub source: PathBuf,
    #[arg(long, default_value_t = false)]
    pub no_migrate: bool,
    #[arg(long, default_value_t = false)]
    pub no_index: bool,
    #[arg(long, default_value_t = false)]
    pub force_reindex: bool,
}

#[derive(Args, Debug)]
pub struct IndexArgs {
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    pub query: String,
    #[arg(long = "deck")]
    pub deck_names: Vec<String>,
    #[arg(long = "tag")]
    pub tags: Vec<String>,
    #[arg(short = 'n', long, default_value_t = 10)]
    pub limit: usize,
    #[arg(long, default_value_t = false)]
    pub chunks: bool,
    #[arg(long, default_value_t = false)]
    pub semantic: bool,
    #[arg(long, default_value_t = false)]
    pub fts: bool,
    #[arg(long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct TopicsArgs {
    #[command(subcommand)]
    pub command: TopicsCommand,
}

#[derive(Subcommand, Debug)]
pub enum TopicsCommand {
    Tree(TopicsTreeArgs),
    Load(TopicsLoadArgs),
    Label(TopicsLabelArgs),
}

#[derive(Args, Debug)]
pub struct TopicsTreeArgs {
    #[arg(long)]
    pub root_path: Option<String>,
}

#[derive(Args, Debug)]
pub struct TopicsLoadArgs {
    #[arg(long)]
    pub file: PathBuf,
}

#[derive(Args, Debug)]
pub struct TopicsLabelArgs {
    #[arg(long)]
    pub file: Option<PathBuf>,
    #[arg(long, default_value_t = 0.6)]
    pub min_confidence: f32,
}

#[derive(Args, Debug)]
pub struct CoverageArgs {
    pub topic: String,
    #[arg(long, default_value_t = false)]
    pub no_subtree: bool,
}

#[derive(Args, Debug)]
pub struct GapsArgs {
    pub topic: String,
    #[arg(long, default_value_t = 1)]
    pub min_coverage: i64,
}

#[derive(Args, Debug)]
pub struct WeakNotesArgs {
    pub topic: String,
    #[arg(short = 'n', long, default_value_t = 20)]
    pub limit: i64,
}

#[derive(Args, Debug)]
pub struct DuplicatesArgs {
    #[arg(long, default_value_t = 0.92)]
    pub threshold: f64,
    #[arg(long, default_value_t = 50)]
    pub max: usize,
    #[arg(long = "deck")]
    pub deck_names: Vec<String>,
    #[arg(long = "tag")]
    pub tags: Vec<String>,
    #[arg(long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    pub file: PathBuf,
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

#[derive(Args, Debug)]
pub struct ValidateArgs {
    pub file: PathBuf,
    #[arg(short, long, default_value_t = false)]
    pub quality: bool,
}

#[derive(Args, Debug)]
pub struct ObsidianSyncArgs {
    pub vault: PathBuf,
    #[arg(long = "source-dirs", value_delimiter = ',')]
    pub source_dirs: Vec<String>,
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

#[derive(Args, Debug)]
pub struct TagAuditArgs {
    pub file: PathBuf,
    #[arg(short, long, default_value_t = false)]
    pub fix: bool,
}

#[derive(Args, Debug)]
pub struct CardloopArgs {
    #[command(subcommand)]
    pub command: CardloopCommand,
}

#[derive(Subcommand, Debug)]
pub enum CardloopCommand {
    /// Scan cards and populate the work queue
    Scan(CardloopScanArgs),
    /// Show score dashboard and progress
    Status(CardloopStatusArgs),
    /// Get next work item(s) from the queue
    Next(CardloopNextArgs),
    /// Mark item(s) as fixed/skipped/wontfix
    Resolve(CardloopResolveArgs),
    /// Show recent progression history
    Log(CardloopLogArgs),
}

#[derive(Args, Debug)]
pub struct CardloopScanArgs {
    /// Which loop to scan: audit, generation, or all
    #[arg(long, default_value = "all")]
    pub loop_kind: String,
    /// Path to card registry SQLite database
    #[arg(long)]
    pub registry: PathBuf,
    /// Path to Anki collection file (.anki2) for FSRS-based retention analysis
    #[arg(long)]
    pub anki_collection: Option<PathBuf>,
    /// Run LLM-powered quality review on all cards
    #[arg(long, default_value_t = false)]
    pub llm_review: bool,
    /// Detect semantic duplicates using Qdrant (requires database + Qdrant running)
    #[arg(long, default_value_t = false)]
    pub detect_duplicates: bool,
    /// Similarity threshold for duplicate detection
    #[arg(long, default_value_t = 0.82)]
    pub dup_threshold: f64,
}

#[derive(Args, Debug)]
pub struct CardloopStatusArgs {
    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct CardloopNextArgs {
    /// Number of items to show
    #[arg(short = 'n', long, default_value_t = 1)]
    pub count: usize,
    /// Filter by loop kind: audit or generation
    #[arg(long)]
    pub loop_kind: Option<String>,
    /// Filter by cluster ID (slug or kind:batch)
    #[arg(long)]
    pub cluster: Option<String>,
}

#[derive(Args, Debug)]
pub struct CardloopResolveArgs {
    /// Work item ID (prefix match supported)
    pub id: String,
    /// Resolution status: fixed, skipped, or wontfix
    #[arg(long, default_value = "fixed")]
    pub status: String,
    /// Attestation: explain what was done
    #[arg(long)]
    pub attest: Option<String>,
    /// Path to card registry SQLite database (used for verification gate)
    #[arg(long)]
    pub registry: Option<std::path::PathBuf>,
}

#[derive(Args, Debug)]
pub struct CardloopLogArgs {
    /// Number of recent events to show
    #[arg(short = 'n', long, default_value_t = 10)]
    pub count: usize,
}
