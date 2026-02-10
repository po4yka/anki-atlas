-- Anki Atlas Initial Schema
-- Migration: 001_initial_schema

-- Decks table
CREATE TABLE IF NOT EXISTS decks (
    deck_id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_name TEXT,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decks_name ON decks (name);
CREATE INDEX IF NOT EXISTS idx_decks_parent ON decks (parent_name);

-- Notes table (primary unit)
CREATE TABLE IF NOT EXISTS notes (
    note_id BIGINT PRIMARY KEY,
    model_id BIGINT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    fields_json JSONB NOT NULL,
    raw_fields TEXT,
    normalized_text TEXT NOT NULL,
    mtime BIGINT NOT NULL,
    usn INTEGER NOT NULL,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_notes_fts ON notes
    USING gin (to_tsvector('english', normalized_text));
CREATE INDEX IF NOT EXISTS idx_notes_model ON notes (model_id);
CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_notes_mtime ON notes (mtime);
CREATE INDEX IF NOT EXISTS idx_notes_deleted ON notes (deleted_at) WHERE deleted_at IS NOT NULL;

-- Cards table (children of notes)
CREATE TABLE IF NOT EXISTS cards (
    card_id BIGINT PRIMARY KEY,
    note_id BIGINT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    deck_id BIGINT NOT NULL REFERENCES decks(deck_id) ON DELETE CASCADE,
    ord INTEGER NOT NULL DEFAULT 0,
    due INTEGER,
    ivl INTEGER DEFAULT 0,
    ease INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    reps INTEGER DEFAULT 0,
    queue INTEGER DEFAULT 0,
    type INTEGER DEFAULT 0,
    mtime BIGINT NOT NULL,
    usn INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cards_note ON cards (note_id);
CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards (deck_id);
CREATE INDEX IF NOT EXISTS idx_cards_mtime ON cards (mtime);
CREATE INDEX IF NOT EXISTS idx_cards_ivl ON cards (ivl);
CREATE INDEX IF NOT EXISTS idx_cards_lapses ON cards (lapses);

-- Card stats (derived from revlog)
CREATE TABLE IF NOT EXISTS card_stats (
    card_id BIGINT PRIMARY KEY REFERENCES cards(card_id) ON DELETE CASCADE,
    reviews INTEGER DEFAULT 0,
    avg_ease REAL,
    fail_rate REAL,
    last_review_at TIMESTAMPTZ,
    total_time_ms BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Note models (Anki note types)
CREATE TABLE IF NOT EXISTS models (
    model_id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    fields JSONB NOT NULL,
    templates JSONB NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sync metadata
CREATE TABLE IF NOT EXISTS sync_metadata (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Topics (user-defined taxonomy) - for later milestones
CREATE TABLE IF NOT EXISTS topics (
    topic_id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topics_path ON topics (path);

-- Note-topic assignments - for later milestones
CREATE TABLE IF NOT EXISTS note_topics (
    note_id BIGINT REFERENCES notes(note_id) ON DELETE CASCADE,
    topic_id INTEGER REFERENCES topics(topic_id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (note_id, topic_id)
);

-- Helper function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_decks_updated_at') THEN
        CREATE TRIGGER update_decks_updated_at BEFORE UPDATE ON decks
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_notes_updated_at') THEN
        CREATE TRIGGER update_notes_updated_at BEFORE UPDATE ON notes
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_cards_updated_at') THEN
        CREATE TRIGGER update_cards_updated_at BEFORE UPDATE ON cards
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_card_stats_updated_at') THEN
        CREATE TRIGGER update_card_stats_updated_at BEFORE UPDATE ON card_stats
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_models_updated_at') THEN
        CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_sync_metadata_updated_at') THEN
        CREATE TRIGGER update_sync_metadata_updated_at BEFORE UPDATE ON sync_metadata
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_topics_updated_at') THEN
        CREATE TRIGGER update_topics_updated_at BEFORE UPDATE ON topics
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Insert initial sync metadata
INSERT INTO sync_metadata (key, value) VALUES
    ('schema_version', '"001"'),
    ('normalization_version', '"1"'),
    ('last_sync_at', 'null')
ON CONFLICT (key) DO NOTHING;
