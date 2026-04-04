-- Knowledge graph: concept and topic relationship edges.
-- Supports similarity, prerequisite, related, cross-reference, and specialization links.

DO $$ BEGIN
    CREATE TYPE edge_type AS ENUM (
        'similar',
        'prerequisite',
        'related',
        'cross_reference',
        'specialization'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE edge_source AS ENUM (
        'embedding',
        'tag_cooccurrence',
        'review_inference',
        'wikilink',
        'taxonomy',
        'manual'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS concept_edges (
    source_note_id BIGINT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    target_note_id BIGINT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    edge_type edge_type NOT NULL,
    edge_source edge_source NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_note_id, target_note_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_concept_edges_target ON concept_edges (target_note_id);
CREATE INDEX IF NOT EXISTS idx_concept_edges_type ON concept_edges (edge_type);

CREATE TABLE IF NOT EXISTS topic_edges (
    source_topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    target_topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    edge_type edge_type NOT NULL,
    edge_source edge_source NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_topic_id, target_topic_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_topic_edges_target ON topic_edges (target_topic_id);
