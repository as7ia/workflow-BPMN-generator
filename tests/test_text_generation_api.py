"""Mock-only API tests for English-text-to-BPMN generation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.bpmn import BpmnLayoutError
from app.domain import ProcessModel
from app.llm import OllamaExtractionError, OllamaSettings
from app.main import create_app

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(create_app())


def _model() -> ProcessModel:
    data = json.loads((FIXTURES / "linear_process.json").read_text(encoding="utf-8"))
    return ProcessModel.model_validate(data)


def test_successful_text_pipeline_always_requires_review() -> None:
    settings = OllamaSettings()
    with (
        patch(
            "app.api.v1.routes.text_generation.OllamaSettings.from_env",
            return_value=settings,
        ),
        patch(
            "app.api.v1.routes.text_generation.extract_process",
            return_value=_model(),
        ) as extraction,
        patch(
            "app.api.v1.routes.text_generation.generate_bpmn",
            return_value="<bpmn:definitions />",
        ) as generation,
    ):
        response = client.post(
            "/api/v1/generate-from-text",
            json={"description": "An employee reviews a request and completes it."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "bpmn_xml": "<bpmn:definitions />",
        "review_required": True,
    }
    extraction.assert_called_once_with(
        "An employee reviews a request and completes it.", settings=settings
    )
    assert generation.call_args.kwargs == {"timeout_seconds": 10.0}


@pytest.mark.parametrize("description", ["", "   ", "A" * 12_001])
def test_invalid_or_excessive_description_is_rejected_before_llm(
    description: str,
) -> None:
    with patch("app.api.v1.routes.text_generation.extract_process") as extraction:
        response = client.post(
            "/api/v1/generate-from-text", json={"description": description}
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_validation_failed"
    extraction.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (OllamaExtractionError("model_unavailable", "private model path"), 503, "ollama_model_unavailable"),
        (OllamaExtractionError("ollama_unavailable", "http://private-host"), 503, "ollama_unavailable"),
        (OllamaExtractionError("ollama_timeout", "internal timeout"), 504, "ollama_timeout"),
        (OllamaExtractionError("unsupported_description", "model explanation"), 422, "unsupported_description"),
        (OllamaExtractionError("malformed_output", "raw sensitive response"), 502, "llm_output_invalid"),
        (OllamaExtractionError("schema_validation_failed", "raw content", details={"issues": [{"location": ["nodes"], "type": "missing"}]}), 502, "llm_schema_invalid"),
        (OllamaExtractionError("graph_validation_failed", "raw content", details={"issues": [{"code": "unknown_scope", "location": ["nodes", 0]}]}), 502, "llm_graph_invalid"),
    ],
)
def test_ollama_failures_have_stable_safe_responses(
    error: OllamaExtractionError, status: int, code: str
) -> None:
    with (
        patch(
            "app.api.v1.routes.text_generation.OllamaSettings.from_env",
            return_value=OllamaSettings(),
        ),
        patch(
            "app.api.v1.routes.text_generation.extract_process", side_effect=error
        ),
        patch("app.api.v1.routes.text_generation.generate_bpmn") as generation,
    ):
        response = client.post(
            "/api/v1/generate-from-text",
            json={"description": "An employee reviews a request."},
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    for sensitive in ("private", "raw content", "raw sensitive", "model explanation"):
        assert sensitive not in response.text
    generation.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (BpmnLayoutError("worker_unavailable", "C:\\private\\node"), 503, "worker_unavailable"),
        (BpmnLayoutError("layout_timeout", "secret", details={"timeout_seconds": 10}), 504, "layout_timeout"),
        (BpmnLayoutError("invalid_di", "TOKEN=secret", details={"stderr": "secret"}), 502, "layout_failed"),
    ],
)
def test_layout_failures_are_sanitized(
    error: BpmnLayoutError, status: int, code: str
) -> None:
    with (
        patch(
            "app.api.v1.routes.text_generation.OllamaSettings.from_env",
            return_value=OllamaSettings(),
        ),
        patch(
            "app.api.v1.routes.text_generation.extract_process", return_value=_model()
        ),
        patch(
            "app.api.v1.routes.text_generation.generate_bpmn", side_effect=error
        ),
    ):
        response = client.post(
            "/api/v1/generate-from-text",
            json={"description": "An employee reviews a request."},
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret" not in response.text.lower()
    assert "private" not in response.text.lower()
    assert "stderr" not in response.text.lower()


def test_invalid_environment_configuration_is_sanitized() -> None:
    with (
        patch(
            "app.api.v1.routes.text_generation.OllamaSettings.from_env",
            side_effect=ValueError("OLLAMA_BASE_URL contains private value"),
        ),
        patch("app.api.v1.routes.text_generation.extract_process") as extraction,
    ):
        response = client.post(
            "/api/v1/generate-from-text",
            json={"description": "An employee reviews a request."},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ollama_configuration_invalid"
    assert "private value" not in response.text
    extraction.assert_not_called()


def test_existing_json_endpoint_health_and_docs_are_unchanged() -> None:
    with patch(
        "app.api.v1.routes.generation.generate_bpmn",
        return_value="<existing-json-endpoint />",
    ):
        response = client.post(
            "/api/v1/generate-from-json",
            json=json.loads(
                (FIXTURES / "linear_process.json").read_text(encoding="utf-8")
            ),
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "bpmn_xml": "<existing-json-endpoint />",
    }
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
