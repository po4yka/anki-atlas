pub mod config;
pub mod error;
pub mod logging;
pub mod types;

// Re-export key items at crate root for ergonomics.
pub use config::{get_settings, Settings};
pub use error::{AnkiAtlasError, Result};
pub use types::{CardId, DeckName, Language, NoteId, SlugStr};
