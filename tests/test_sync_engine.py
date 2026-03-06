"""Tests for the sync engine package."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from packages.anki.sync.engine import SyncEngine, SyncResult
from packages.anki.sync.progress import ProgressTracker, SyncPhase, SyncProgress
from packages.anki.sync.recovery import CardRecovery, CardTransaction, RollbackAction
from packages.anki.sync.state import CardState, StateDB

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# CardState
# ---------------------------------------------------------------------------


class TestCardState:
    def test_creation(self) -> None:
        state = CardState(slug="test-card", content_hash="abc123")
        assert state.slug == "test-card"
        assert state.content_hash == "abc123"

    def test_frozen(self) -> None:
        state = CardState(slug="test", content_hash="hash")
        with pytest.raises(AttributeError):
            state.slug = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        state = CardState(slug="s", content_hash="h")
        assert state.anki_guid is None
        assert state.note_type == ""
        assert state.source_path == ""
        assert state.synced_at == 0.0


# ---------------------------------------------------------------------------
# StateDB
# ---------------------------------------------------------------------------


class TestStateDB:
    @pytest.fixture
    def db(self, tmp_path: Path) -> StateDB:
        return StateDB(tmp_path / "test.db")

    def test_get_nonexistent(self, db: StateDB) -> None:
        assert db.get("nonexistent") is None

    def test_upsert_and_get(self, db: StateDB) -> None:
        state = CardState(slug="card-1", content_hash="hash1", note_type="basic")
        db.upsert(state)
        result = db.get("card-1")
        assert result is not None
        assert result.slug == "card-1"
        assert result.content_hash == "hash1"
        assert result.note_type == "basic"

    def test_upsert_updates(self, db: StateDB) -> None:
        db.upsert(CardState(slug="card-1", content_hash="hash1"))
        db.upsert(CardState(slug="card-1", content_hash="hash2"))
        result = db.get("card-1")
        assert result is not None
        assert result.content_hash == "hash2"

    def test_delete(self, db: StateDB) -> None:
        db.upsert(CardState(slug="card-1", content_hash="hash1"))
        db.delete("card-1")
        assert db.get("card-1") is None

    def test_delete_nonexistent(self, db: StateDB) -> None:
        db.delete("nonexistent")  # Should not raise

    def test_get_all(self, db: StateDB) -> None:
        db.upsert(CardState(slug="b-card", content_hash="h1"))
        db.upsert(CardState(slug="a-card", content_hash="h2"))
        results = db.get_all()
        assert len(results) == 2
        assert results[0].slug == "a-card"
        assert results[1].slug == "b-card"

    def test_get_by_source(self, db: StateDB) -> None:
        db.upsert(CardState(slug="c1", content_hash="h1", source_path="notes/a.md"))
        db.upsert(CardState(slug="c2", content_hash="h2", source_path="notes/a.md"))
        db.upsert(CardState(slug="c3", content_hash="h3", source_path="notes/b.md"))
        results = db.get_by_source("notes/a.md")
        assert len(results) == 2
        assert all(r.source_path == "notes/a.md" for r in results)

    def test_wal_mode(self, tmp_path: Path) -> None:
        db = StateDB(tmp_path / "wal.db")
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] == "wal"
        db.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        with StateDB(tmp_path / "ctx.db") as db:
            db.upsert(CardState(slug="c1", content_hash="h1"))
            assert db.get("c1") is not None


# ---------------------------------------------------------------------------
# SyncPhase
# ---------------------------------------------------------------------------


class TestSyncPhase:
    def test_values(self) -> None:
        assert SyncPhase.INITIALIZING == "initializing"
        assert SyncPhase.COMPLETED == "completed"
        assert SyncPhase.FAILED == "failed"

    def test_all_phases(self) -> None:
        phases = list(SyncPhase)
        assert len(phases) == 7


# ---------------------------------------------------------------------------
# SyncProgress
# ---------------------------------------------------------------------------


class TestSyncProgress:
    def test_creation(self) -> None:
        progress = SyncProgress(session_id="s1", phase=SyncPhase.SCANNING)
        assert progress.session_id == "s1"
        assert progress.phase == SyncPhase.SCANNING

    def test_frozen(self) -> None:
        progress = SyncProgress(session_id="s1", phase=SyncPhase.SCANNING)
        with pytest.raises(AttributeError):
            progress.session_id = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        progress = SyncProgress(session_id="s1", phase=SyncPhase.INITIALIZING)
        assert progress.total_notes == 0
        assert progress.notes_processed == 0
        assert progress.cards_created == 0
        assert progress.errors == 0


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_auto_session_id(self) -> None:
        tracker = ProgressTracker()
        snap = tracker.snapshot()
        assert len(snap.session_id) == 12

    def test_custom_session_id(self) -> None:
        tracker = ProgressTracker(session_id="my-session")
        assert tracker.snapshot().session_id == "my-session"

    def test_set_phase(self) -> None:
        tracker = ProgressTracker()
        tracker.set_phase(SyncPhase.SCANNING)
        assert tracker.snapshot().phase == SyncPhase.SCANNING

    def test_set_total(self) -> None:
        tracker = ProgressTracker()
        tracker.set_total(100)
        assert tracker.snapshot().total_notes == 100

    def test_increment(self) -> None:
        tracker = ProgressTracker()
        tracker.increment("notes_processed", 5)
        tracker.increment("cards_created", 3)
        snap = tracker.snapshot()
        assert snap.notes_processed == 5
        assert snap.cards_created == 3

    def test_increment_invalid_stat(self) -> None:
        tracker = ProgressTracker()
        with pytest.raises(ValueError, match="Invalid stat"):
            tracker.increment("invalid_stat")

    def test_complete_success(self) -> None:
        tracker = ProgressTracker()
        tracker.complete(success=True)
        assert tracker.snapshot().phase == SyncPhase.COMPLETED

    def test_complete_failure(self) -> None:
        tracker = ProgressTracker()
        tracker.complete(success=False)
        assert tracker.snapshot().phase == SyncPhase.FAILED

    def test_progress_pct(self) -> None:
        tracker = ProgressTracker()
        assert tracker.progress_pct == 0.0
        tracker.set_total(10)
        tracker.increment("notes_processed", 5)
        assert tracker.progress_pct == pytest.approx(50.0)

    def test_progress_pct_zero_total(self) -> None:
        tracker = ProgressTracker()
        assert tracker.progress_pct == 0.0


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_creation(self) -> None:
        result = SyncResult(cards_created=5, cards_updated=3)
        assert result.cards_created == 5
        assert result.cards_updated == 3

    def test_frozen(self) -> None:
        result = SyncResult()
        with pytest.raises(AttributeError):
            result.cards_created = 1  # type: ignore[misc]

    def test_defaults(self) -> None:
        result = SyncResult()
        assert result.cards_created == 0
        assert result.cards_skipped == 0
        assert result.duration_ms == 0


# ---------------------------------------------------------------------------
# SyncEngine
# ---------------------------------------------------------------------------


class TestSyncEngine:
    @pytest.fixture
    def db(self, tmp_path: Path) -> StateDB:
        return StateDB(tmp_path / "engine.db")

    def test_init(self, db: StateDB) -> None:
        engine = SyncEngine(db)
        assert engine.state_db is db
        assert engine.progress is not None

    def test_sync_lifecycle(self, db: StateDB) -> None:
        db.upsert(CardState(slug="c1", content_hash="h1"))
        engine = SyncEngine(db)
        result = engine.sync()
        assert isinstance(result, SyncResult)
        assert result.duration_ms >= 0

    def test_sync_dry_run(self, db: StateDB) -> None:
        engine = SyncEngine(db)
        result = engine.sync(dry_run=True)
        assert isinstance(result, SyncResult)
        snap = engine.progress.snapshot()
        assert snap.phase == SyncPhase.COMPLETED

    def test_sync_progress_phases(self, db: StateDB) -> None:
        tracker = ProgressTracker(session_id="test")
        engine = SyncEngine(db, progress=tracker)
        engine.sync()
        snap = tracker.snapshot()
        assert snap.phase == SyncPhase.COMPLETED


# ---------------------------------------------------------------------------
# RollbackAction
# ---------------------------------------------------------------------------


class TestRollbackAction:
    def test_creation(self) -> None:
        action = RollbackAction(action_type="delete", target_id="card-1")
        assert action.action_type == "delete"
        assert action.target_id == "card-1"

    def test_frozen(self) -> None:
        action = RollbackAction(action_type="delete", target_id="card-1")
        with pytest.raises(AttributeError):
            action.action_type = "create"  # type: ignore[misc]

    def test_defaults(self) -> None:
        action = RollbackAction(action_type="delete", target_id="card-1")
        assert action.succeeded is False
        assert action.error == ""


# ---------------------------------------------------------------------------
# CardTransaction
# ---------------------------------------------------------------------------


class TestCardTransaction:
    def test_add_and_commit(self) -> None:
        txn = CardTransaction()
        txn.add_rollback("create", "card-1")
        txn.add_rollback("create", "card-2")
        txn.commit()
        assert txn.rollback() == ()

    def test_rollback(self) -> None:
        txn = CardTransaction()
        txn.add_rollback("create", "card-1")
        txn.add_rollback("update", "card-2")
        actions = txn.rollback()
        assert len(actions) == 2
        assert actions[0].target_id == "card-2"  # reversed order
        assert actions[1].target_id == "card-1"

    def test_context_manager_commit(self) -> None:
        with CardTransaction() as txn:
            txn.add_rollback("create", "card-1")
            txn.commit()
        # No rollback should happen

    def test_context_manager_exception(self) -> None:
        try:
            with CardTransaction() as txn:
                txn.add_rollback("create", "card-1")
                msg = "test error"
                raise RuntimeError(msg)
        except RuntimeError:
            pass
        # Rollback should have been triggered (committed=False after exit)


# ---------------------------------------------------------------------------
# CardRecovery
# ---------------------------------------------------------------------------


class TestCardRecovery:
    @pytest.fixture
    def db(self, tmp_path: Path) -> StateDB:
        return StateDB(tmp_path / "recovery.db")

    def test_find_orphaned(self, db: StateDB) -> None:
        recovery = CardRecovery(db)
        db_slugs = frozenset({"a", "b", "c"})
        anki_slugs = frozenset({"b", "c", "d"})
        in_db_not_anki, in_anki_not_db = recovery.find_orphaned(db_slugs, anki_slugs)
        assert in_db_not_anki == frozenset({"a"})
        assert in_anki_not_db == frozenset({"d"})

    def test_find_orphaned_no_orphans(self, db: StateDB) -> None:
        recovery = CardRecovery(db)
        slugs = frozenset({"a", "b"})
        in_db, in_anki = recovery.find_orphaned(slugs, slugs)
        assert in_db == frozenset()
        assert in_anki == frozenset()

    def test_find_stale(self, db: StateDB) -> None:
        old_time = time.time() - (60 * 86400)  # 60 days ago
        db.upsert(CardState(slug="old", content_hash="h1", synced_at=old_time))
        db.upsert(CardState(slug="new", content_hash="h2", synced_at=time.time()))
        recovery = CardRecovery(db)
        stale = recovery.find_stale(max_age_days=30)
        assert len(stale) == 1
        assert stale[0].slug == "old"

    def test_find_stale_excludes_zero(self, db: StateDB) -> None:
        db.upsert(CardState(slug="unsynced", content_hash="h1", synced_at=0.0))
        recovery = CardRecovery(db)
        stale = recovery.find_stale(max_age_days=30)
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# Existing imports preserved
# ---------------------------------------------------------------------------


class TestExistingImports:
    def test_sync_anki_collection_import(self) -> None:
        from packages.anki.sync import sync_anki_collection

        assert callable(sync_anki_collection)

    def test_sync_service_import(self) -> None:
        from packages.anki.sync import SyncService

        assert SyncService is not None

    def test_sync_stats_import(self) -> None:
        from packages.anki.sync import SyncStats

        assert SyncStats is not None
