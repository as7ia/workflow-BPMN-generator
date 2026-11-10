"""Tests for the health endpoint and API documentation."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_swagger_documentation_is_available() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()
