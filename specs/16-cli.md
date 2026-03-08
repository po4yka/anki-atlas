# Spec: crate `cli`

## Source Reference
Python: `apps/cli/__init__.py` + `apps/cli/generate.py`, `apps/cli/validate.py`, `apps/cli/obsidian.py`, `apps/cli/tags.py`

## Purpose
Command-line interface for anki-atlas providing all user-facing operations: sync, migrate, index, search, topics, coverage, gaps, duplicates, generate, validate, obsidian-sync, and tag-audit. Uses clap derive for argument parsing, `anyhow` for error handling, and `tokio` for async runtime. Output is formatted for terminal using `comfy-table` or plain text with ANSI colors via `console`.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
llm = { path = "../llm" }
obsidian = { path = "../obsidian" }
rag = { path = "../rag" }
generator = { path = "../generator" }
jobs = { path = "../jobs" }
# Additional workspace crates as needed:
# anki-sync, indexer, search, analytics, taxonomy, validation, card

anyhow = "1"
clap = { version = "4", features = ["derive", "env"] }
comfy-table = "7"
console = "0.15"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Public API

### CLI Entry Point (`src/main.rs`)

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "anki-atlas", about = "Searchable hybrid index for Anki collections")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
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
```

### Subcommand Args (`src/args.rs`)

```rust
use std::path::PathBuf;

#[derive(clap::Args)]
pub struct SyncArgs {
    /// Path to collection.anki2 file.
    #[arg(short, long)]
    pub source: String,

    /// Run database migrations before sync.
    #[arg(long, default_value_t = true)]
    pub migrate: bool,

    /// Index notes to vector database after sync.
    #[arg(long, default_value_t = true)]
    pub index: bool,

    /// Force re-embedding all notes.
    #[arg(long, default_value_t = false)]
    pub force_reindex: bool,
}

#[derive(clap::Args)]
pub struct IndexArgs {
    /// Force re-embedding all notes.
    #[arg(short, long, default_value_t = false)]
    pub force: bool,
}

#[derive(clap::Args)]
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

#[derive(clap::Args)]
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

#[derive(clap::Args)]
pub struct CoverageArgs {
    /// Topic path (e.g., programming/python).
    pub topic: String,

    /// Include child topics.
    #[arg(long, default_value_t = true)]
    pub subtree: bool,
}

#[derive(clap::Args)]
pub struct GapsArgs {
    /// Topic path.
    pub topic: String,

    /// Minimum notes for coverage.
    #[arg(short, long, default_value_t = 1)]
    pub min_coverage: usize,
}

#[derive(clap::Args)]
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

#[derive(clap::Args)]
pub struct GenerateArgs {
    /// Path to an Obsidian markdown note.
    pub file: PathBuf,

    /// Preview without generating.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

#[derive(clap::Args)]
pub struct ValidateArgs {
    /// File with card front/back (--- separated).
    pub file: PathBuf,

    /// Run quality assessment.
    #[arg(short, long, default_value_t = false)]
    pub quality: bool,
}

#[derive(clap::Args)]
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

#[derive(clap::Args)]
pub struct TagAuditArgs {
    /// File with tags, one per line.
    pub file: PathBuf,

    /// Show normalized tags.
    #[arg(short, long, default_value_t = false)]
    pub fix: bool,
}
```

### Command Handlers (`src/commands/*.rs`)

Each command is implemented as an async function that:
1. Parses/validates arguments.
2. Constructs service dependencies.
3. Calls the appropriate package API.
4. Formats and prints results as tables or structured text.
5. Returns `anyhow::Result<()>`, printing errors via `eprintln!`.

```rust
// Example: src/commands/sync.rs
pub async fn run(args: &SyncArgs) -> anyhow::Result<()>;

// Example: src/commands/search.rs
pub async fn run(args: &SearchArgs) -> anyhow::Result<()>;
```

### Module structure

```
src/
  main.rs          -- CLI entry, clap parse, dispatch
  args.rs          -- All arg structs
  commands/
    mod.rs
    sync.rs
    migrate.rs
    index.rs
    search.rs
    topics.rs
    coverage.rs
    gaps.rs
    duplicates.rs
    generate.rs
    validate.rs
    obsidian_sync.rs
    tag_audit.rs
  output.rs        -- Table formatting helpers
```

## Internal Details

### Error Handling Pattern
- Each command handler wraps operations in a closure-style error handler.
- On error: log with tracing, print user-friendly message to stderr, exit with code 1.
- Uses `anyhow::Context` for adding context to errors.

### Output Formatting
- Tables use `comfy-table` crate with columns for structured data (search results, sync stats, coverage).
- Colors via `console` crate: green for success, red for errors, yellow for warnings, cyan for labels.
- Verbose mode in search shows detailed per-result scores.

### Async Runtime
- `main` uses `#[tokio::main]` with multi-threaded runtime.
- All command handlers are async.
- Tracing subscriber configured at startup with env-filter.

### Search Command
- Builds `SearchFilters` from deck/tag options.
- Calls `SearchService::search` with limit, semantic_only, fts_only flags.
- Fetches note details for result enrichment.
- Displays table with rank, note ID, score, sources, preview, tags.

### Duplicates Command
- Constructs `DuplicateDetector`, calls `find_duplicates`.
- Shows summary stats then cluster details.
- In verbose mode shows all duplicates; otherwise top match + count.

## Acceptance Criteria
- [ ] `clap` parses all 12 subcommands correctly
- [ ] `version` command prints version string
- [ ] `sync` command validates source path exists before proceeding
- [ ] `sync` command runs migrations, sync, and index in sequence
- [ ] `migrate` command calls migration function and reports results
- [ ] `index` command respects `--force` flag
- [ ] `search` command accepts query, deck, tag, top, semantic, fts, verbose flags
- [ ] `search` command displays results in table format
- [ ] `topics` command loads taxonomy from file or database
- [ ] `coverage` command displays metrics table with coverage data
- [ ] `gaps` command separates missing and undercovered topics
- [ ] `duplicates` command respects threshold, max, deck, tag, verbose options
- [ ] `generate` command parses Obsidian note and shows preview
- [ ] `validate` command runs validation pipeline and optionally quality scoring
- [ ] `obsidian-sync` command discovers and parses vault notes
- [ ] `tag-audit` command validates tags and shows normalized versions with `--fix`
- [ ] Error handling prints user-friendly messages and exits with code 1
- [ ] `--help` works for all commands and subcommands
- [ ] `make check` equivalent passes (clippy, fmt, test)
