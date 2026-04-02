use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::error::CardloopError;
use crate::models::ProgressionEvent;

/// Append-only JSONL log for progression events.
pub struct ProgressionLog {
    path: PathBuf,
}

impl ProgressionLog {
    /// Create a log backed by the given file path.
    /// Creates parent directories if needed.
    pub fn open(path: &Path) -> Result<Self, CardloopError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    /// Append a single event as a JSON line.
    pub fn append(&self, event: &ProgressionEvent) -> Result<(), CardloopError> {
        let line = serde_json::to_string(event)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Read the most recent `n` events (tail of the file).
    pub fn read_recent(&self, n: usize) -> Result<Vec<ProgressionEvent>, CardloopError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut events: Vec<ProgressionEvent> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<ProgressionEvent>(trimmed) {
                Ok(event) => events.push(event),
                Err(_) => continue, // Skip malformed lines
            }
        }

        // Return the last n events
        if events.len() > n {
            events = events.split_off(events.len() - n);
        }

        Ok(events)
    }

    /// Total number of events in the log.
    pub fn count(&self) -> Result<usize, CardloopError> {
        if !self.path.exists() {
            return Ok(0);
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let count = reader
            .lines()
            .map_while(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .count();

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_event(action: &str) -> ProgressionEvent {
        ProgressionEvent {
            timestamp: Utc::now(),
            action: action.to_string(),
            item_ids: vec!["item-1".into()],
            actor: "agent".into(),
            note: None,
            scores_before: None,
            scores_after: None,
        }
    }

    #[test]
    fn append_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progression.jsonl");
        let log = ProgressionLog::open(&path).unwrap();

        log.append(&make_event("scan")).unwrap();
        log.append(&make_event("resolve")).unwrap();
        log.append(&make_event("skip")).unwrap();

        assert_eq!(log.count().unwrap(), 3);

        let recent = log.read_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].action, "resolve");
        assert_eq!(recent[1].action, "skip");
    }

    #[test]
    fn read_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.jsonl");
        let log = ProgressionLog::open(&path).unwrap();

        let events = log.read_recent(10).unwrap();
        assert!(events.is_empty());
        assert_eq!(log.count().unwrap(), 0);
    }

    #[test]
    fn read_recent_more_than_available() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progression.jsonl");
        let log = ProgressionLog::open(&path).unwrap();

        log.append(&make_event("scan")).unwrap();

        let events = log.read_recent(100).unwrap();
        assert_eq!(events.len(), 1);
    }
}
