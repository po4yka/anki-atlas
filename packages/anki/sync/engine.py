"""Sync engine orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from packages.anki.sync.progress import ProgressTracker, SyncPhase

if TYPE_CHECKING:
    from packages.anki.sync.state import StateDB

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Result of a sync operation."""

    cards_created: int = 0
    cards_updated: int = 0
    cards_deleted: int = 0
    cards_skipped: int = 0
    errors: int = 0
    duration_ms: int = 0


class SyncEngine:
    """Orchestrate sync lifecycle.

    The engine manages phases of a sync operation, delegating actual
    card operations to injected components via the state database.
    """

    def __init__(
        self,
        state_db: StateDB,
        *,
        progress: ProgressTracker | None = None,
    ) -> None:
        self._state_db = state_db
        self._progress = progress or ProgressTracker()
        self._cards_created = 0
        self._cards_updated = 0
        self._cards_deleted = 0
        self._cards_skipped = 0
        self._errors = 0

    @property
    def state_db(self) -> StateDB:
        """Access the state database."""
        return self._state_db

    @property
    def progress(self) -> ProgressTracker:
        """Access the progress tracker."""
        return self._progress

    def sync(self, *, dry_run: bool = False) -> SyncResult:
        """Run sync lifecycle.

        Args:
            dry_run: If True, scan but do not apply changes.

        Returns:
            SyncResult with operation counts.
        """
        start = time.monotonic()
        session = self._progress.snapshot().session_id
        log.info("sync.started", session_id=session, dry_run=dry_run)

        try:
            self._progress.set_phase(SyncPhase.SCANNING)
            self._scan()

            if not dry_run:
                self._progress.set_phase(SyncPhase.APPLYING)
                self._apply()

            self._progress.complete(success=True)
        except Exception:
            self._progress.complete(success=False)
            self._errors += 1
            log.exception("sync.failed", session_id=session)
            raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result = SyncResult(
            cards_created=self._cards_created,
            cards_updated=self._cards_updated,
            cards_deleted=self._cards_deleted,
            cards_skipped=self._cards_skipped,
            errors=self._errors,
            duration_ms=elapsed_ms,
        )
        log.info("sync.completed", session_id=session, result=result)
        return result

    def _scan(self) -> None:
        """Scan for changes (subclasses or callers populate state)."""
        existing = self._state_db.get_all()
        self._progress.set_total(len(existing))
        log.debug("sync.scan_complete", card_count=len(existing))

    def _apply(self) -> None:
        """Apply changes (subclasses override for actual operations)."""
        log.debug("sync.apply_complete")
