use crate::models::*;
use common::error::Result;
use std::path::{Path, PathBuf};

/// Read an Anki collection from a SQLite database file.
pub struct AnkiReader {
    collection_path: PathBuf,
}

impl AnkiReader {
    pub fn new(_collection_path: impl AsRef<Path>) -> Result<Self> {
        todo!()
    }

    pub fn open(&mut self) -> Result<()> {
        todo!()
    }

    pub fn close(&mut self) {
        todo!()
    }

    pub fn read_collection(&self) -> Result<AnkiCollection> {
        todo!()
    }

    pub fn read_decks(&self) -> Result<Vec<AnkiDeck>> {
        todo!()
    }

    pub fn read_models(&self) -> Result<Vec<AnkiModel>> {
        todo!()
    }

    pub fn read_notes(&self, _models: &[AnkiModel]) -> Result<Vec<AnkiNote>> {
        todo!()
    }

    pub fn read_cards(&self) -> Result<Vec<AnkiCard>> {
        todo!()
    }

    pub fn read_revlog(&self) -> Result<Vec<AnkiRevlogEntry>> {
        todo!()
    }

    pub fn compute_card_stats(&self) -> Result<Vec<CardStats>> {
        todo!()
    }
}

impl Drop for AnkiReader {
    fn drop(&mut self) {
        // cleanup
    }
}

/// Convenience function: open, read, close.
pub fn read_anki_collection(_path: impl AsRef<Path>) -> Result<AnkiCollection> {
    todo!()
}
