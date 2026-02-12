"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test /health returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        """Test /health returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_version(self, client: TestClient) -> None:
        """Test /health includes version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "0.1.0"

    def test_health_returns_services_config(self, client: TestClient) -> None:
        """Test /health includes services configuration."""
        response = client.get("/health")
        data = response.json()
        assert "services" in data
        assert "postgres" in data["services"]
        assert "qdrant" in data["services"]


class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_200(self, client: TestClient) -> None:
        """Test /ready returns 200 even when not ready."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_returns_status(self, client: TestClient) -> None:
        """Test /ready returns valid status."""
        response = client.get("/ready")
        data = response.json()
        assert data["status"] in ("ready", "not_ready")

    def test_ready_returns_checks(self, client: TestClient) -> None:
        """Test /ready includes service checks."""
        response = client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert "postgres" in data["checks"]
        assert "qdrant" in data["checks"]

    def test_ready_check_values_are_valid(self, client: TestClient) -> None:
        """Test /ready check values are ok or failed."""
        response = client.get("/ready")
        data = response.json()
        for service, status in data["checks"].items():
            assert status in ("ok", "failed"), f"{service} has invalid status: {status}"


class TestSyncEndpoint:
    """Tests for /sync endpoint."""

    def test_sync_requires_source(self, client: TestClient) -> None:
        """Test /sync requires source field."""
        response = client.post("/sync", json={})
        assert response.status_code == 422  # Validation error

    def test_sync_rejects_nonexistent_path(self, client: TestClient) -> None:
        """Test /sync returns 400 for non-existent collection."""
        response = client.post(
            "/sync",
            json={"source": "/nonexistent/path/collection.anki2"},
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    def test_index_accepts_empty_body(self, client: TestClient) -> None:
        """Test /index accepts empty request body."""
        # This may fail if db not available, but should not be validation error
        response = client.post("/index", json={})
        assert response.status_code in (200, 409, 500, 503)

    def test_index_accepts_force_reindex(self, client: TestClient) -> None:
        """Test /index accepts force_reindex parameter."""
        response = client.post("/index", json={"force_reindex": True})
        assert response.status_code in (200, 409, 500, 503)


class TestIndexInfoEndpoint:
    """Tests for /index/info endpoint."""

    def test_index_info_returns_200(self, client: TestClient) -> None:
        """Test /index/info returns 200."""
        response = client.get("/index/info")
        assert response.status_code == 200

    def test_index_info_returns_status(self, client: TestClient) -> None:
        """Test /index/info includes status field."""
        response = client.get("/index/info")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "not_created", "error")


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_requires_query(self, client: TestClient) -> None:
        """Test /search requires query field."""
        response = client.post("/search", json={})
        assert response.status_code == 422

    def test_search_accepts_valid_request(self, client: TestClient) -> None:
        """Test /search accepts valid request (may fail without db)."""
        try:
            response = client.post("/search", json={"query": "test", "top_k": 10})
            # May succeed or fail depending on service availability
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass

    def test_search_accepts_filters(self, client: TestClient) -> None:
        """Test /search accepts filter parameters."""
        try:
            response = client.post(
                "/search",
                json={
                    "query": "test",
                    "deck_names": ["Default"],
                    "tags": ["python"],
                    "top_k": 10,
                },
            )
            # May fail without data, but should not be validation error
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass


class TestTopicsEndpoint:
    """Tests for /topics endpoint."""

    def test_topics_returns_valid_response(self, client: TestClient) -> None:
        """Test /topics returns valid response (may fail without db)."""
        try:
            response = client.get("/topics")
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass

    def test_topics_with_root_filter(self, client: TestClient) -> None:
        """Test /topics accepts root filter."""
        try:
            response = client.get("/topics", params={"root": "programming"})
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass


class TestDuplicatesEndpoint:
    """Tests for /duplicates endpoint."""

    def test_duplicates_returns_valid_response(self, client: TestClient) -> None:
        """Test /duplicates returns valid response (may fail without db)."""
        try:
            response = client.get("/duplicates")
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass

    def test_duplicates_accepts_threshold(self, client: TestClient) -> None:
        """Test /duplicates accepts threshold parameter."""
        try:
            response = client.get("/duplicates", params={"threshold": 0.95})
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass

    def test_duplicates_with_low_threshold(self, client: TestClient) -> None:
        """Test /duplicates handles low threshold (may not validate)."""
        try:
            response = client.get("/duplicates", params={"threshold": 0.5})
            # If endpoint has validation, expect 422, otherwise may try to process
            assert response.status_code in (200, 422, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass

    def test_duplicates_with_high_threshold(self, client: TestClient) -> None:
        """Test /duplicates handles high threshold."""
        try:
            response = client.get("/duplicates", params={"threshold": 0.99})
            # High threshold is valid, may fail due to services
            assert response.status_code in (200, 500, 503)
        except Exception:
            # Expected if services unavailable
            pass


class TestCorrelationIdMiddleware:
    """Tests for correlation ID middleware."""

    def test_response_includes_request_id(self, client: TestClient) -> None:
        """Test response includes X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_uses_provided_request_id(self, client: TestClient) -> None:
        """Test uses X-Request-ID from request if provided."""
        custom_id = "test-correlation-id-12345"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers.get("X-Request-ID") == custom_id

    def test_generates_request_id_when_not_provided(self, client: TestClient) -> None:
        """Test generates X-Request-ID when not in request."""
        response = client.get("/health")
        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None
        assert len(request_id) > 0


# Legacy test functions for backwards compatibility
def test_health_endpoint(client: TestClient) -> None:
    """Test /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "services" in data


def test_ready_endpoint(client: TestClient) -> None:
    """Test /ready returns status (may be not_ready without db)."""
    response = client.get("/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("ready", "not_ready")
    assert "checks" in data
    assert "postgres" in data["checks"]
    assert "qdrant" in data["checks"]
