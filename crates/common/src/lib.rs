pub mod config;
pub mod error;
pub mod logging;
pub mod types;

/// Assert that one or more types implement `Send + Sync` at compile time.
///
/// ```rust,ignore
/// common::assert_send_sync!(MyType, MyOtherType);
/// ```
#[macro_export]
macro_rules! assert_send_sync {
    ($($t:ty),+ $(,)?) => {
        $(
            const _: () = {
                fn _assert<T: Send + Sync + ?Sized>() {}
                fn _check() { _assert::<$t>(); }
            };
        )+
    };
}

// Re-export key items at crate root for ergonomics.
pub use config::{
    ApiSettings, DatabaseSettings, EmbeddingProviderKind, EmbeddingSettings, JobSettings,
    Quantization, RerankSettings, Settings,
};
pub use error::{AnkiAtlasError, Result};
pub use types::{CardId, DeckId, DeckName, Language, ModelId, NoteId, SlugStr};
