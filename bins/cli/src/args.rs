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
