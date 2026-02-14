"""Async background jobs backed by arq and Redis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from arq.connections import ArqRedis, RedisSettings, create_pool

from packages.common.config import Settings, get_settings
from packages.common.logging import get_logger

logger = get_logger(module=__name__)

JobType = Literal["sync", "index"]
JobStatus = Literal[
    "queued",
    "scheduled",
    "running",
    "retrying",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
]

JOB_KEY_PREFIX = "ankiatlas:job:"
TERMINAL_STATUSES: set[JobStatus] = {"succeeded", "failed", "cancelled"}


class JobBackendUnavailableError(RuntimeError):
    """Raised when Redis/arq backend is unavailable."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def build_redis_settings(redis_url: str) -> RedisSettings:
    """Build arq RedisSettings from redis URL."""
    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", "rediss"}:
        raise ValueError("redis_url must use redis:// or rediss://")

    database = int(parsed.path.lstrip("/") or "0")
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=database,
        username=parsed.username,
        password=parsed.password,
        ssl=parsed.scheme == "rediss",
        conn_timeout=1,
        conn_retries=1,
        conn_retry_delay=1,
    )


@dataclass
class JobRecord:
    """Persisted metadata for an async job."""

    job_id: str
    job_type: JobType
    status: JobStatus
    payload: dict[str, Any]
    progress: float = 0.0
    message: str | None = None
    attempts: int = 0
    max_retries: int = 3
    cancel_requested: bool = False
    created_at: datetime | None = None
    scheduled_for: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "payload": self.payload,
            "progress": self.progress,
            "message": self.message,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        return cls(
            job_id=str(data["job_id"]),
            job_type=data["job_type"],
            status=data["status"],
            payload=dict(data.get("payload", {})),
            progress=float(data.get("progress", 0.0)),
            message=data.get("message"),
            attempts=int(data.get("attempts", 0)),
            max_retries=int(data.get("max_retries", 3)),
            cancel_requested=bool(data.get("cancel_requested", False)),
            created_at=_parse_datetime(data.get("created_at")),
            scheduled_for=_parse_datetime(data.get("scheduled_for")),
            started_at=_parse_datetime(data.get("started_at")),
            finished_at=_parse_datetime(data.get("finished_at")),
            result=data.get("result"),
            error=data.get("error"),
        )


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


async def save_job_record(redis: Any, record: JobRecord, ttl_seconds: int) -> None:
    """Persist a job record in Redis."""
    await redis.set(
        _job_key(record.job_id),
        json.dumps(record.to_dict()),
        ex=ttl_seconds,
    )


async def load_job_record(redis: Any, job_id: str) -> JobRecord | None:
    """Load a job record from Redis."""
    raw = await redis.get(_job_key(job_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return JobRecord.from_dict(data)


class ArqJobManager:
    """Queue and inspect background jobs using arq."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._redis: ArqRedis | None = None

    async def connect(self) -> ArqRedis:
        """Connect to Redis/arq pool if needed."""
        if self._redis is None:
            try:
                redis_settings = build_redis_settings(self.settings.redis_url)
                self._redis = await create_pool(redis_settings)
            except Exception as exc:
                raise JobBackendUnavailableError(f"Failed to connect to Redis: {exc}") from exc
        return self._redis

    async def close(self) -> None:
        """Close Redis pool."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def enqueue_sync_job(
        self,
        payload: dict[str, Any],
        run_at: datetime | None = None,
    ) -> JobRecord:
        """Enqueue a sync job."""
        return await self._enqueue_job("sync", "job_sync", payload, run_at)

    async def enqueue_index_job(
        self,
        payload: dict[str, Any],
        run_at: datetime | None = None,
    ) -> JobRecord:
        """Enqueue an index job."""
        return await self._enqueue_job("index", "job_index", payload, run_at)

    async def _enqueue_job(
        self,
        job_type: JobType,
        task_name: str,
        payload: dict[str, Any],
        run_at: datetime | None = None,
    ) -> JobRecord:
        redis = await self.connect()
        now = _utcnow()
        normalized_run_at = run_at
        if normalized_run_at and normalized_run_at.tzinfo is None:
            normalized_run_at = normalized_run_at.replace(tzinfo=UTC)
        job_id = str(uuid4())

        status: JobStatus = (
            "scheduled"
            if normalized_run_at is not None and normalized_run_at > now
            else "queued"
        )
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status=status,
            payload=payload,
            created_at=now,
            scheduled_for=normalized_run_at,
            max_retries=self.settings.job_max_retries,
            message="Job accepted",
        )

        await save_job_record(redis, record, self.settings.job_result_ttl_seconds)

        job = await redis.enqueue_job(
            task_name,
            job_id=job_id,
            payload=payload,
            _job_id=job_id,
            _queue_name=self.settings.job_queue_name,
            _defer_until=normalized_run_at,
        )
        if job is None:
            raise RuntimeError(f"Failed to enqueue job '{job_id}'")

        logger.info(
            "job_enqueued",
            job_id=job_id,
            job_type=job_type,
            run_at=normalized_run_at.isoformat() if normalized_run_at else None,
        )
        return record

    async def get_job(self, job_id: str) -> JobRecord | None:
        """Get current job metadata."""
        redis = await self.connect()
        return await load_job_record(redis, job_id)

    async def cancel_job(self, job_id: str) -> JobRecord | None:
        """Request cancellation of a queued/running job."""
        redis = await self.connect()
        record = await load_job_record(redis, job_id)
        if record is None:
            return None

        if record.status in TERMINAL_STATUSES:
            return record

        record.cancel_requested = True
        if record.status in {"queued", "scheduled", "retrying"}:
            record.status = "cancelled"
            record.progress = 100.0
            record.finished_at = _utcnow()
            record.message = "Cancelled before execution"
        else:
            record.status = "cancel_requested"
            record.message = "Cancellation requested"

        await save_job_record(redis, record, self.settings.job_result_ttl_seconds)
        logger.info("job_cancel_requested", job_id=job_id, status=record.status)
        return record


_job_manager: ArqJobManager | None = None


async def get_job_manager(settings: Settings | None = None) -> ArqJobManager:
    """Get cached job manager."""
    global _job_manager
    if _job_manager is None:
        _job_manager = ArqJobManager(settings)
    return _job_manager


async def close_job_manager() -> None:
    """Close cached job manager resources."""
    global _job_manager
    if _job_manager is not None:
        await _job_manager.close()
        _job_manager = None
