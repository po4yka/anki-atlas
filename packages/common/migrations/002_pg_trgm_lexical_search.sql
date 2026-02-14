-- Enable trigram search for typo-tolerant lexical matching
-- Migration: 002_pg_trgm_lexical_search

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Improves fuzzy similarity and autocomplete fallback on note text
CREATE INDEX IF NOT EXISTS idx_notes_normalized_text_trgm
    ON notes USING gin (normalized_text gin_trgm_ops)
    WHERE deleted_at IS NULL;

-- Helpful for deck-name fuzzy filtering paths
CREATE INDEX IF NOT EXISTS idx_decks_name_trgm
    ON decks USING gin (name gin_trgm_ops);

INSERT INTO sync_metadata (key, value)
VALUES ('schema_version', '"002"')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

