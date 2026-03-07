"""Progress tracking for sync operations."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

from packages.common.logging import get_logger

logger = get_logger(module=__name__)


class SyncPhase(StrEnum):
    """Phases of a sync operation."""

    INITIALIZING = "initializing"
    INDEXING = "indexing"
    SCANNING = "scanning"
    GENERATING = "generating"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SyncProgress:
    """Snapshot of sync progress at a point in time."""

    session_id: str
    phase: SyncPhase
    total_notes: int = 0
    notes_processed: int = 0
    cards_created: int = 0
    cards_updated: int = 0
    cards_deleted: int = 0
    errors: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0


_VALID_STATS = frozenset(
    {
        "notes_processed",
        "cards_created",
        "cards_updated",
        "cards_deleted",
        "errors",
    }
)


class ProgressTracker:
    """Track sync progress (thread-safe)."""

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._phase = SyncPhase.INITIALIZING
        self._total_notes = 0
        self._notes_processed = 0
        self._cards_created = 0
        self._cards_updated = 0
        self._cards_deleted = 0
        self._errors = 0
        self._started_at = time.time()
        self._updated_at = self._started_at

    def set_phase(self, phase: SyncPhase) -> None:
        """Set the current sync phase."""
        with self._lock:
            self._phase = phase
            self._updated_at = time.time()
        logger.info("sync.phase_changed", session_id=self._session_id, phase=phase.value)

    def set_total(self, total: int) -> None:
        """Set total number of notes to process."""
        with self._lock:
            self._total_notes = total
            self._updated_at = time.time()

    def increment(self, stat: str, count: int = 1) -> None:
        """Increment a progress stat."""
        if stat not in _VALID_STATS:
            msg = f"Invalid stat: {stat}. Must be one of {_VALID_STATS}"
            raise ValueError(msg)
        with self._lock:
            current = getattr(self, f"_{stat}")
            setattr(self, f"_{stat}", current + count)
            self._updated_at = time.time()

    def snapshot(self) -> SyncProgress:
        """Get a frozen snapshot of current progress."""
        with self._lock:
            return SyncProgress(
                session_id=self._session_id,
                phase=self._phase,
                total_notes=self._total_notes,
                notes_processed=self._notes_processed,
                cards_created=self._cards_created,
                cards_updated=self._cards_updated,
                cards_deleted=self._cards_deleted,
                errors=self._errors,
                started_at=self._started_at,
                updated_at=self._updated_at,
            )

    def complete(self, *, success: bool = True) -> None:
        """Mark sync as completed or failed."""
        phase = SyncPhase.COMPLETED if success else SyncPhase.FAILED
        self.set_phase(phase)

    @property
    def progress_pct(self) -> float:
        """Get progress percentage (0.0 to 100.0)."""
        with self._lock:
            if self._total_notes == 0:
                return 0.0
            return min(100.0, (self._notes_processed / self._total_notes) * 100.0)
