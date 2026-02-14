"""arq worker tasks for sync/index jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common.config import get_settings
from packages.common.database import run_migrations
from packages.common.logging import get_logger
from packages.jobs.service import JobRecord, load_job_record, save_job_record

logger = get_logger(module=__name__)


def _now() -> datetime:
    return datetime.now(UTC)


async def _load_or_raise(redis: Any, job_id: str) -> JobRecord:
    record = await load_job_record(redis, job_id)
    if record is None:
        raise ValueError(f"Job not found: {job_id}")
    return record


async def _set_status(
    redis: Any,
    *,
    job_id: str,
    status: str,
    progress: float | None = None,
    message: str | None = None,
    attempts: int | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
    finished: bool = False,
) -> JobRecord:
    settings = get_settings()
    record = await _load_or_raise(redis, job_id)
    record.status = status  # type: ignore[assignment]
    if progress is not None:
        record.progress = max(0.0, min(100.0, progress))
    if message is not None:
        record.message = message
    if attempts is not None:
        record.attempts = attempts
    if error is not None:
        record.error = error
    if result is not None:
        record.result = result
    if finished:
        record.finished_at = _now()
        record.progress = 100.0

    await save_job_record(redis, record, settings.job_result_ttl_seconds)
    return record


async def _is_cancelled(redis: Any, job_id: str) -> bool:
    record = await _load_or_raise(redis, job_id)
    return record.cancel_requested or record.status in {"cancel_requested", "cancelled"}


async def _cancel_if_requested(redis: Any, job_id: str) -> bool:
    if not await _is_cancelled(redis, job_id):
        return False
    await _set_status(
        redis,
        job_id=job_id,
        status="cancelled",
        message="Cancelled",
        finished=True,
    )
    return True


async def job_sync(ctx: dict[str, Any], job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Background task: sync Anki collection and optionally index vectors."""
    from packages.anki.sync import sync_anki_collection
    from packages.indexer.service import EmbeddingModelChanged, index_all_notes

    settings = get_settings()
    redis = ctx["redis"]
    attempt = int(ctx.get("job_try", 1))

    await _set_status(
        redis,
        job_id=job_id,
        status="running",
        progress=5.0,
        attempts=attempt,
        message="Starting sync job",
    )

    try:
        if await _cancel_if_requested(redis, job_id):
            return {"cancelled": True}

        source = str(payload.get("source", ""))
        source_path = Path(source).expanduser().resolve()
        if not source or not source_path.exists():
            raise FileNotFoundError(f"Collection not found: {source_path}")

        if bool(payload.get("run_migrations", True)):
            await _set_status(
                redis,
                job_id=job_id,
                status="running",
                progress=15.0,
                message="Running migrations",
            )
            await run_migrations(settings)

        if await _cancel_if_requested(redis, job_id):
            return {"cancelled": True}

        await _set_status(
            redis,
            job_id=job_id,
            status="running",
            progress=40.0,
            message="Syncing collection to PostgreSQL",
        )
        sync_stats = await sync_anki_collection(source_path, settings=settings)

        result: dict[str, Any] = {
            "sync": {
                "decks_upserted": sync_stats.decks_upserted,
                "models_upserted": sync_stats.models_upserted,
                "notes_upserted": sync_stats.notes_upserted,
                "notes_deleted": sync_stats.notes_deleted,
                "cards_upserted": sync_stats.cards_upserted,
                "card_stats_upserted": sync_stats.card_stats_upserted,
                "duration_ms": sync_stats.duration_ms,
            }
        }

        if bool(payload.get("index", True)):
            if await _cancel_if_requested(redis, job_id):
                return {"cancelled": True}

            await _set_status(
                redis,
                job_id=job_id,
                status="running",
                progress=75.0,
                message="Indexing vectors",
            )
            try:
                index_stats = await index_all_notes(
                    settings=settings,
                    force_reindex=bool(payload.get("force_reindex", False)),
                )
                result["index"] = {
                    "notes_processed": index_stats.notes_processed,
                    "notes_embedded": index_stats.notes_embedded,
                    "notes_skipped": index_stats.notes_skipped,
                    "notes_deleted": index_stats.notes_deleted,
                    "errors": index_stats.errors,
                }
            except EmbeddingModelChanged as exc:
                result["index"] = {
                    "notes_processed": 0,
                    "notes_embedded": 0,
                    "notes_skipped": 0,
                    "notes_deleted": 0,
                    "errors": [str(exc)],
                }

        await _set_status(
            redis,
            job_id=job_id,
            status="succeeded",
            message="Job completed",
            result=result,
            finished=True,
        )
        return result
    except Exception as exc:
        retryable = not isinstance(exc, FileNotFoundError)
        if retryable and attempt < settings.job_max_retries:
            await _set_status(
                redis,
                job_id=job_id,
                status="retrying",
                progress=0.0,
                error=str(exc),
                message=f"Retrying after error: {type(exc).__name__}",
                attempts=attempt,
            )
            logger.exception("job_sync_failed_retrying", job_id=job_id, attempt=attempt)
            raise
        else:
            await _set_status(
                redis,
                job_id=job_id,
                status="failed",
                error=str(exc),
                message=f"Job failed: {type(exc).__name__}",
                attempts=attempt,
                finished=True,
            )
            logger.exception("job_sync_failed_terminal", job_id=job_id, attempt=attempt)
            return {"error": str(exc)}


async def job_index(ctx: dict[str, Any], job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Background task: index notes to vector store."""
    from packages.indexer.service import index_all_notes

    settings = get_settings()
    redis = ctx["redis"]
    attempt = int(ctx.get("job_try", 1))

    await _set_status(
        redis,
        job_id=job_id,
        status="running",
        progress=10.0,
        attempts=attempt,
        message="Starting indexing job",
    )

    try:
        if await _cancel_if_requested(redis, job_id):
            return {"cancelled": True}

        stats = await index_all_notes(
            settings=settings,
            force_reindex=bool(payload.get("force_reindex", False)),
        )
        result = {
            "notes_processed": stats.notes_processed,
            "notes_embedded": stats.notes_embedded,
            "notes_skipped": stats.notes_skipped,
            "notes_deleted": stats.notes_deleted,
            "errors": stats.errors,
        }

        await _set_status(
            redis,
            job_id=job_id,
            status="succeeded",
            message="Job completed",
            result=result,
            finished=True,
        )
        return result
    except Exception as exc:
        if attempt < settings.job_max_retries:
            await _set_status(
                redis,
                job_id=job_id,
                status="retrying",
                progress=0.0,
                error=str(exc),
                message=f"Retrying after error: {type(exc).__name__}",
                attempts=attempt,
            )
            logger.exception("job_index_failed_retrying", job_id=job_id, attempt=attempt)
            raise
        else:
            await _set_status(
                redis,
                job_id=job_id,
                status="failed",
                error=str(exc),
                message=f"Job failed: {type(exc).__name__}",
                attempts=attempt,
                finished=True,
            )
            logger.exception("job_index_failed_terminal", job_id=job_id, attempt=attempt)
            return {"error": str(exc)}
