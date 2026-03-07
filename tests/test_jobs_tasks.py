"""Tests for packages/jobs/tasks.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.jobs.service import JobRecord
from packages.jobs.tasks import _cancel_if_requested, _is_cancelled, _load_or_raise, _set_status


def _make_record(**overrides: Any) -> JobRecord:
    """Create a test JobRecord with sensible defaults."""
    defaults: dict[str, Any] = {
        "job_id": "test-job-1",
        "job_type": "sync",
        "status": "queued",
        "payload": {},
        "progress": 0.0,
        "cancel_requested": False,
    }
    defaults.update(overrides)
    return JobRecord(**defaults)


class TestLoadOrRaise:
    """Tests for _load_or_raise."""

    async def test_returns_record(self) -> None:
        record = _make_record()
        redis = AsyncMock()
        with patch(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        ):
            result = await _load_or_raise(redis, "test-job-1")
        assert result is record

    async def test_raises_on_missing(self) -> None:
        redis = AsyncMock()
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=None),
            ),
            pytest.raises(ValueError, match="Job not found"),
        ):
            await _load_or_raise(redis, "missing")


class TestSetStatus:
    """Tests for _set_status helper."""

    async def test_updates_status(self) -> None:
        record = _make_record()
        redis = AsyncMock()
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=record),
            ),
            patch(
                "packages.jobs.tasks.save_job_record",
                AsyncMock(),
            ) as mock_save,
            patch("packages.jobs.tasks.get_settings", MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600))),
        ):
            result = await _set_status(
                redis,
                job_id="test-job-1",
                status="running",
                progress=50.0,
                message="halfway",
            )

        assert result.status == "running"
        assert result.progress == 50.0
        assert result.message == "halfway"
        mock_save.assert_awaited_once()

    async def test_clamps_progress(self) -> None:
        record = _make_record()
        redis = AsyncMock()
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=record),
            ),
            patch("packages.jobs.tasks.save_job_record", AsyncMock()),
            patch("packages.jobs.tasks.get_settings", MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600))),
        ):
            result = await _set_status(
                redis, job_id="test-job-1", status="running", progress=150.0
            )
        assert result.progress == 100.0

    async def test_finished_sets_progress_100(self) -> None:
        record = _make_record()
        redis = AsyncMock()
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=record),
            ),
            patch("packages.jobs.tasks.save_job_record", AsyncMock()),
            patch("packages.jobs.tasks.get_settings", MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600))),
        ):
            result = await _set_status(
                redis, job_id="test-job-1", status="succeeded", finished=True
            )
        assert result.progress == 100.0
        assert result.finished_at is not None


class TestIsCancelled:
    """Tests for _is_cancelled."""

    async def test_not_cancelled(self) -> None:
        record = _make_record(status="running", cancel_requested=False)
        with patch(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        ):
            assert not await _is_cancelled(AsyncMock(), "test-job-1")

    async def test_cancel_requested_flag(self) -> None:
        record = _make_record(status="running", cancel_requested=True)
        with patch(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        ):
            assert await _is_cancelled(AsyncMock(), "test-job-1")

    async def test_cancel_requested_status(self) -> None:
        record = _make_record(status="cancel_requested")
        with patch(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        ):
            assert await _is_cancelled(AsyncMock(), "test-job-1")


class TestCancelIfRequested:
    """Tests for _cancel_if_requested."""

    async def test_not_cancelled(self) -> None:
        record = _make_record(status="running", cancel_requested=False)
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=record),
            ),
            patch("packages.jobs.tasks.save_job_record", AsyncMock()),
            patch("packages.jobs.tasks.get_settings", MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600))),
        ):
            result = await _cancel_if_requested(AsyncMock(), "test-job-1")
        assert result is False

    async def test_cancelled(self) -> None:
        record = _make_record(status="running", cancel_requested=True)
        with (
            patch(
                "packages.jobs.tasks.load_job_record",
                AsyncMock(return_value=record),
            ),
            patch("packages.jobs.tasks.save_job_record", AsyncMock()),
            patch("packages.jobs.tasks.get_settings", MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600))),
        ):
            result = await _cancel_if_requested(AsyncMock(), "test-job-1")
        assert result is True


class TestJobSync:
    """Tests for job_sync task."""

    @staticmethod
    def _patch_task_infra(monkeypatch: pytest.MonkeyPatch, record: JobRecord) -> None:
        """Patch common infrastructure for job tasks."""
        monkeypatch.setattr(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        )
        monkeypatch.setattr("packages.jobs.tasks.save_job_record", AsyncMock())
        monkeypatch.setattr(
            "packages.jobs.tasks.get_settings",
            MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600, job_max_retries=3)),
        )

    async def test_sync_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from packages.jobs.tasks import job_sync

        col_file = tmp_path / "collection.anki2"
        col_file.touch()

        record = _make_record(status="queued")
        mock_stats = MagicMock(
            decks_upserted=2,
            models_upserted=1,
            notes_upserted=5,
            notes_deleted=0,
            cards_upserted=5,
            card_stats_upserted=3,
            duration_ms=100,
        )

        self._patch_task_infra(monkeypatch, record)

        ctx: dict[str, Any] = {"redis": AsyncMock(), "job_try": 1}
        payload = {"source": str(col_file), "index": False}

        with (
            patch(
                "packages.anki.sync.sync_anki_collection",
                AsyncMock(return_value=mock_stats),
            ),
            patch(
                "packages.jobs.tasks.run_migrations",
                AsyncMock(return_value=MagicMock(applied=[])),
            ),
        ):
            result = await job_sync(ctx, "test-job-1", payload)

        assert "sync" in result
        assert result["sync"]["notes_upserted"] == 5

    async def test_sync_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from packages.jobs.tasks import job_sync

        record = _make_record(status="queued")
        self._patch_task_infra(monkeypatch, record)

        ctx: dict[str, Any] = {"redis": AsyncMock(), "job_try": 1}
        payload = {"source": "/nonexistent/path.anki2", "index": False}

        result = await job_sync(ctx, "test-job-1", payload)
        assert "error" in result

    async def test_sync_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from packages.jobs.tasks import job_sync

        record = _make_record(status="running", cancel_requested=True)
        self._patch_task_infra(monkeypatch, record)

        ctx: dict[str, Any] = {"redis": AsyncMock(), "job_try": 1}
        payload = {"source": "/any", "index": False}

        result = await job_sync(ctx, "test-job-1", payload)
        assert result.get("cancelled") is True


class TestJobIndex:
    """Tests for job_index task."""

    async def test_index_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from packages.jobs.tasks import job_index

        record = _make_record(job_type="index", status="queued")
        mock_stats = MagicMock(
            notes_processed=10,
            notes_embedded=8,
            notes_skipped=2,
            notes_deleted=0,
            errors=[],
        )

        monkeypatch.setattr(
            "packages.jobs.tasks.load_job_record",
            AsyncMock(return_value=record),
        )
        monkeypatch.setattr("packages.jobs.tasks.save_job_record", AsyncMock())
        monkeypatch.setattr(
            "packages.jobs.tasks.get_settings",
            MagicMock(return_value=MagicMock(job_result_ttl_seconds=3600, job_max_retries=3)),
        )

        ctx: dict[str, Any] = {"redis": AsyncMock(), "job_try": 1}

        with patch(
            "packages.indexer.service.index_all_notes",
            AsyncMock(return_value=mock_stats),
        ):
            result = await job_index(ctx, "test-job-1", {})

        assert result["notes_processed"] == 10
        assert result["notes_embedded"] == 8
