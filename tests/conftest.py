"""Pytest configuration and fixtures."""

import json
import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_collection(temp_dir: Path) -> Path:
    """Create a minimal Anki collection for testing.

    Creates a SQLite database with the same schema as Anki's collection.anki2.
    """
    db_path = temp_dir / "collection.anki2"
    conn = sqlite3.connect(str(db_path))

    # Create minimal Anki schema
    conn.executescript(
        """
        -- Collection metadata
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            scm INTEGER NOT NULL,
            ver INTEGER NOT NULL,
            dty INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            ls INTEGER NOT NULL,
            conf TEXT NOT NULL,
            models TEXT NOT NULL,
            decks TEXT NOT NULL,
            dconf TEXT NOT NULL,
            tags TEXT NOT NULL
        );

        -- Notes
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            guid TEXT NOT NULL,
            mid INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            tags TEXT NOT NULL,
            flds TEXT NOT NULL,
            sfld INTEGER NOT NULL,
            csum INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );

        -- Cards
        CREATE TABLE cards (
            id INTEGER PRIMARY KEY,
            nid INTEGER NOT NULL,
            did INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            type INTEGER NOT NULL,
            queue INTEGER NOT NULL,
            due INTEGER NOT NULL,
            ivl INTEGER NOT NULL,
            factor INTEGER NOT NULL,
            reps INTEGER NOT NULL,
            lapses INTEGER NOT NULL,
            left INTEGER NOT NULL,
            odue INTEGER NOT NULL,
            odid INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );

        -- Review log
        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY,
            cid INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            ease INTEGER NOT NULL,
            ivl INTEGER NOT NULL,
            lastIvl INTEGER NOT NULL,
            factor INTEGER NOT NULL,
            time INTEGER NOT NULL,
            type INTEGER NOT NULL
        );
        """
    )

    # Sample decks
    decks = {
        "1": {
            "id": 1,
            "name": "Default",
            "mod": 1700000000,
            "usn": -1,
            "lrnToday": [0, 0],
            "revToday": [0, 0],
            "newToday": [0, 0],
            "timeToday": [0, 0],
            "collapsed": False,
            "desc": "",
            "dyn": 0,
            "conf": 1,
        },
        "1234567890": {
            "id": 1234567890,
            "name": "Programming::Python",
            "mod": 1700000000,
            "usn": -1,
            "lrnToday": [0, 0],
            "revToday": [0, 0],
            "newToday": [0, 0],
            "timeToday": [0, 0],
            "collapsed": False,
            "desc": "",
            "dyn": 0,
            "conf": 1,
        },
    }

    # Sample model (note type)
    models = {
        "1234567891": {
            "id": 1234567891,
            "name": "Basic",
            "type": 0,
            "mod": 1700000000,
            "usn": -1,
            "sortf": 0,
            "did": 1,
            "tmpls": [
                {
                    "name": "Card 1",
                    "ord": 0,
                    "qfmt": "{{Front}}",
                    "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
                    "did": None,
                    "bqfmt": "",
                    "bafmt": "",
                }
            ],
            "flds": [
                {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20},
                {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20},
            ],
            "css": "",
            "latexPre": "",
            "latexPost": "",
            "latexsvg": False,
            "req": [[0, "all", [0]]],
        }
    }

    # Insert collection metadata
    conn.execute(
        """
        INSERT INTO col (id, crt, mod, scm, ver, dty, usn, ls, conf, models, decks, dconf, tags)
        VALUES (1, 1700000000, 1700000000, 1700000000, 11, 0, -1, 0, '{}', ?, ?, '{}', '{}')
        """,
        (json.dumps(models), json.dumps(decks)),
    )

    # Insert sample notes
    notes = [
        (
            1000000001,
            "abc123",
            1234567891,
            1700000000,
            -1,
            "python programming",
            "What is a <b>list</b> in Python?\x1fAn ordered, mutable collection of items.",
            0,
            0,
            0,
            "",
        ),
        (
            1000000002,
            "def456",
            1234567891,
            1700000000,
            -1,
            "python",
            "What is a <code>dict</code>?\x1fA key-value mapping data structure.",
            0,
            0,
            0,
            "",
        ),
        (
            1000000003,
            "ghi789",
            1234567891,
            1700000000,
            -1,
            "",
            "{{c1::Lambda}} functions are anonymous.\x1fThey use the lambda keyword.",
            0,
            0,
            0,
            "",
        ),
    ]

    conn.executemany(
        "INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        notes,
    )

    # Insert sample cards
    cards = [
        (2000000001, 1000000001, 1234567890, 0, 1700000000, -1, 2, 2, 100, 21, 2500, 10, 2, 0, 0, 0, 0, ""),
        (2000000002, 1000000002, 1234567890, 0, 1700000000, -1, 2, 2, 50, 14, 2300, 5, 1, 0, 0, 0, 0, ""),
        (2000000003, 1000000003, 1, 0, 1700000000, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ""),
    ]

    conn.executemany(
        "INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses, left, odue, odid, flags, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        cards,
    )

    # Insert sample review log
    revlog = [
        (1700000000001, 2000000001, -1, 3, 1, 0, 2500, 5000, 0),
        (1700000000002, 2000000001, -1, 3, 3, 1, 2500, 3000, 1),
        (1700000000003, 2000000001, -1, 1, 1, 3, 2300, 8000, 1),  # Again = fail
        (1700000000004, 2000000002, -1, 4, 7, 0, 2500, 2000, 0),
    ]

    conn.executemany(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, time, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        revlog,
    )

    conn.commit()
    conn.close()

    return db_path
