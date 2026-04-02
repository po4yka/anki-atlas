mod batch;
pub mod embeddings;
pub mod qdrant;
pub mod service;

pub use service::{
    ChunkForIndexing, IndexProgressCallback, IndexProgressEvent, IndexProgressStage, IndexService,
    IndexStats, MultimodalNoteForIndexing, NoteForIndexing,
};
