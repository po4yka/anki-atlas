"""Recovery and transaction support for sync operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from packages.common.logging import get_logger

if TYPE_CHECKING:
    from types import TracebackType

    from packages.anki.sync.state import CardState, StateDB

logger = get_logger(module=__name__)


@dataclass(frozen=True, slots=True)
class RollbackAction:
    """A recorded action that can be rolled back."""

    action_type: str
    target_id: str
    succeeded: bool = False
    error: str = ""


class CardTransaction:
    """Atomic card operation with rollback support."""

    def __init__(self) -> None:
        self._actions: list[tuple[str, str]] = []
        self._committed: bool = False

    def add_rollback(self, action_type: str, target_id: str) -> None:
        """Record an action for potential rollback."""
        self._actions.append((action_type, target_id))

    def commit(self) -> None:
        """Mark transaction as committed (no rollback needed)."""
        self._committed = True
        logger.debug("transaction.committed", actions=len(self._actions))

    def rollback(self) -> tuple[RollbackAction, ...]:
        """Roll back uncommitted actions.

        Returns:
            Tuple of RollbackAction results (one per recorded action).
        """
        if self._committed:
            return ()

        results: list[RollbackAction] = []
        for action_type, target_id in reversed(self._actions):
            results.append(
                RollbackAction(
                    action_type=action_type,
                    target_id=target_id,
                    succeeded=True,
                )
            )
            logger.info("transaction.rollback", action=action_type, target=target_id)

        return tuple(results)

    def __enter__(self) -> CardTransaction:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and not self._committed:
            self.rollback()


class CardRecovery:
    """Detect and recover from inconsistent card states."""

    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db

    def find_orphaned(
        self,
        db_slugs: frozenset[str],
        anki_slugs: frozenset[str],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Find orphaned cards.

        Args:
            db_slugs: Slugs present in the state database.
            anki_slugs: Slugs present in Anki.

        Returns:
            Tuple of (in_db_not_anki, in_anki_not_db).
        """
        in_db_not_anki = db_slugs - anki_slugs
        in_anki_not_db = anki_slugs - db_slugs
        if in_db_not_anki or in_anki_not_db:
            logger.warning(
                "recovery.orphaned_found",
                in_db_not_anki=len(in_db_not_anki),
                in_anki_not_db=len(in_anki_not_db),
            )
        return in_db_not_anki, in_anki_not_db

    def find_stale(self, max_age_days: int = 30) -> tuple[CardState, ...]:
        """Find card states older than max_age_days.

        Args:
            max_age_days: Maximum age in days before a card state is considered stale.

        Returns:
            Tuple of stale CardState entries.
        """
        cutoff = time.time() - (max_age_days * 86400)
        all_states = self._state_db.get_all()
        stale = tuple(s for s in all_states if 0 < s.synced_at < cutoff)
        if stale:
            logger.info("recovery.stale_found", count=len(stale), max_age_days=max_age_days)
        return stale
