pub mod analyzer;
pub mod error;
pub mod frontmatter;
#[cfg(feature = "fuzzing")]
pub mod fuzzing;
pub mod parser;
pub mod sync;

pub use analyzer::{BrokenLink, VaultAnalyzer, VaultStats};
pub use error::ObsidianError;
pub use frontmatter::{parse_frontmatter, write_frontmatter};
pub use parser::{MAX_FILE_SIZE, ParsedNote, Section, discover_notes, parse_note};
pub use sync::{CardGenerator, GeneratedCardRef, NoteResult, ObsidianSyncWorkflow, SyncResult};
