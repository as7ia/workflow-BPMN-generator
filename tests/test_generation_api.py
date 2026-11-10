"""Fast API tests for JSON-to-BPMN generation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.bpmn import BpmnLayoutError, BpmnSerializationError
from app.domain import ProcessGraphValidationError, ValidationIssue
from app.main import create_app

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(create_app())


def _fixture(name: str = "linear_process.json") -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_valid_request_returns_layouted_xml() -> None:
    with patch(
        "app.api.v1.routes.generation.generate_bpmn",
        return_value="<definitions><BPMNDiagram /></definitions>",
    ) as service:
        response = client.post("/api/v1/generate-from-json", json=_fixture())

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "bpmn_xml": "<definitions><BPMNDiagram /></definitions>",
    }
    assert service.call_args.kwargs == {"timeout_seconds": 10.0}


def test_invalid_json_structure_has_stable_sanitized_error() -> None:
    response = client.post(
        "/api/v1/generate-from-json",
        json={"schema_version": "1.0", "secret": "do-not-return"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "schema_validation_failed"
    assert "do-not-return" not in response.text
    assert all(set(item) == {"location", "message", "type"} for item in body["error"]["details"])


def test_invalid_graph_returns_validation_issues() -> None:
    issue = ValidationIssue(
        code="unknown_sequence_target",
        location=("sequence_flows", 0, "target_ref"),
        message="The target does not exist.",
    )
    with patch(
        "app.api.v1.routes.generation.generate_bpmn",
        side_effect=ProcessGraphValidationError([issue]),
    ):
        response = client.post("/api/v1/generate-from-json", json=_fixture())

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "graph_validation_failed",
        "message": "The process graph is invalid.",
        "details": [
            {
                "code": "unknown_sequence_target",
                "location": ["sequence_flows", 0, "target_ref"],
                "message": "The target does not exist.",
            }
        ],
    }


def test_serialization_failure_is_sanitized() -> None:
    with patch(
        "app.api.v1.routes.generation.generate_bpmn",
        side_effect=BpmnSerializationError("sensitive internal object"),
    ):
        response = client.post("/api/v1/generate-from-json", json=_fixture())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "serialization_failed"
    assert "sensitive internal object" not in response.text


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (BpmnLayoutError("worker_unavailable", "C:\\secret\\node.exe"), 503, "worker_unavailable"),
        (BpmnLayoutError("layout_timeout", "internal", details={"timeout_seconds": 10.0}), 504, "layout_timeout"),
        (BpmnLayoutError("invalid_di", "internal", details={"stderr": "TOKEN=secret"}), 502, "layout_failed"),
        (BpmnLayoutError("expanded_subprocess_unsupported", "internal", details={"subprocess_id": "review"}), 422, "expanded_subprocess_unsupported"),
    ],
)
def test_layout_errors_are_mapped_without_diagnostic_leaks(
    error: BpmnLayoutError, status: int, code: str
) -> None:
    with patch("app.api.v1.routes.generation.generate_bpmn", side_effect=error):
        response = client.post("/api/v1/generate-from-json", json=_fixture())

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret" not in response.text.lower()
    assert "stderr" not in response.text.lower()


def test_request_body_limit_rejects_payload_before_validation() -> None:
    limited_client = TestClient(create_app(max_request_bytes=32))
    response = limited_client.post("/api/v1/generate-from-json", json=_fixture())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_health_and_docs_remain_available() -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
