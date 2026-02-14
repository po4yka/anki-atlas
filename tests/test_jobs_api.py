"""Tests for async jobs API endpoints."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from packages.jobs.service import JobRecord


class FakeJobManager:
    """In-memory fake job manager for API tests."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._counter = 0

    async def enqueue_sync_job(
        self,
        payload: dict[str, object],
        run_at: datetime | None = None,
    ) -> JobRecord:
        return self._create_job("sync", payload, run_at)

    async def enqueue_index_job(
        self,
        payload: dict[str, object],
        run_at: datetime | None = None,
    ) -> JobRecord:
        return self._create_job("index", payload, run_at)

    async def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> JobRecord | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        updated = replace(
            job,
            status="cancelled",
            cancel_requested=True,
            message="Cancelled before execution",
            progress=100.0,
            finished_at=datetime.now(UTC),
        )
        self._jobs[job_id] = updated
        return updated

    def _create_job(
        self,
        job_type: str,
        payload: dict[str, object],
        run_at: datetime | None,
    ) -> JobRecord:
        self._counter += 1
        job_id = f"job-{self._counter}"
        status = "scheduled" if run_at and run_at > datetime.now(UTC) else "queued"
        job = JobRecord(
            job_id=job_id,
            job_type=job_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            payload=dict(payload),
            progress=0.0,
            message="Job accepted",
            attempts=0,
            max_retries=3,
            created_at=datetime.now(UTC),
            scheduled_for=run_at,
        )
        self._jobs[job_id] = job
        return job


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def fake_manager(monkeypatch: pytest.MonkeyPatch) -> FakeJobManager:
    """Patch API to use fake async job manager."""
    manager = FakeJobManager()

    async def _get_job_manager() -> FakeJobManager:
        return manager

    monkeypatch.setattr("apps.api.main.get_job_manager", _get_job_manager)
    return manager


@pytest.mark.usefixtures("fake_manager")
class TestAsyncJobsSyncEndpoint:
    """Tests for POST /jobs/sync."""

    def test_sync_job_requires_source(self, client: TestClient) -> None:
        response = client.post("/jobs/sync", json={})
        assert response.status_code == 422

    def test_sync_job_rejects_nonexistent_path(self, client: TestClient) -> None:
        response = client.post(
            "/jobs/sync",
            json={"source": "/definitely/not/found.anki2"},
        )
        assert response.status_code == 400

    def test_sync_job_accepts_request(self, client: TestClient, temp_dir: Path) -> None:
        source = temp_dir / "collection.anki2"
        source.write_text("dummy")
        response = client.post("/jobs/sync", json={"source": str(source)})
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"].startswith("job-")
        assert data["job_type"] == "sync"
        assert data["status"] == "queued"
        assert data["poll_url"] == f"/jobs/{data['job_id']}"

    def test_sync_job_supports_scheduling(self, client: TestClient, temp_dir: Path) -> None:
        source = temp_dir / "collection.anki2"
        source.write_text("dummy")
        run_at = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
        response = client.post("/jobs/sync", json={"source": str(source), "run_at": run_at})
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "scheduled"
        assert data["scheduled_for"] is not None


@pytest.mark.usefixtures("fake_manager")
class TestAsyncJobsIndexEndpoint:
    """Tests for POST /jobs/index."""

    def test_index_job_accepts_request(self, client: TestClient) -> None:
        response = client.post("/jobs/index", json={"force_reindex": True})
        assert response.status_code == 202
        data = response.json()
        assert data["job_type"] == "index"
        assert data["status"] == "queued"


@pytest.mark.usefixtures("fake_manager")
class TestAsyncJobsStatusAndCancel:
    """Tests for GET /jobs/{id} and POST /jobs/{id}/cancel."""

    def test_get_job_status_returns_404_for_missing(self, client: TestClient) -> None:
        response = client.get("/jobs/does-not-exist")
        assert response.status_code == 404

    def test_get_job_status_returns_job(self, client: TestClient, temp_dir: Path) -> None:
        source = temp_dir / "collection.anki2"
        source.write_text("dummy")
        enqueue = client.post("/jobs/sync", json={"source": str(source)})
        job_id = enqueue.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in {"queued", "scheduled"}

    def test_cancel_job_updates_status(self, client: TestClient, temp_dir: Path) -> None:
        source = temp_dir / "collection.anki2"
        source.write_text("dummy")
        enqueue = client.post("/jobs/sync", json={"source": str(source)})
        job_id = enqueue.json()["job_id"]

        cancel = client.post(f"/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        data = cancel.json()
        assert data["job_id"] == job_id
        assert data["status"] == "cancelled"
        assert data["cancel_requested"] is True
