"""Mock-only tests for bounded local Ollama process extraction."""

import json
import socket
from pathlib import Path
from unittest.mock import patch
from urllib import error as url_error

import pytest

from app.domain import ProcessModel
from app.llm import OllamaExtractionError, OllamaSettings, extract_process

FIXTURES = Path(__file__).parent / "fixtures"


class FakeTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], float, int]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> dict[str, object]:
        self.calls.append((url, payload, timeout_seconds, max_response_bytes))
        return self.response


def _fixture(name: str = "linear_process.json") -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _chat_response(content: object) -> dict[str, object]:
    text = content if isinstance(content, str) else json.dumps(content)
    return {"model": "qwen3:8b", "message": {"role": "assistant", "content": text}, "done": True}


def test_valid_json_extraction_uses_exact_schema_and_one_chat_call() -> None:
    transport = FakeTransport(_chat_response(_fixture()))
    settings = OllamaSettings(timeout_seconds=7, num_predict=1000, num_ctx=4000)

    model = extract_process(
        "An operations employee reviews a request and then completes it.",
        settings=settings,
        transport=transport,
    )

    assert isinstance(model, ProcessModel)
    assert model.id == "linear"
    assert len(transport.calls) == 1
    url, payload, timeout, response_limit = transport.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["format"] == ProcessModel.model_json_schema()
    assert payload["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_predict": 1000,
        "num_ctx": 4000,
    }
    assert timeout == 7
    assert response_limit == settings.max_response_bytes
    assert "BPMN XML" in payload["messages"][0]["content"]


def test_malformed_model_json_is_rejected_without_retry() -> None:
    transport = FakeTransport(_chat_response("{not-json"))

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process("A clerk reviews an application.", transport=transport)

    assert captured.value.code == "malformed_output"
    assert len(transport.calls) == 1


def test_invalid_pydantic_schema_is_rejected() -> None:
    transport = FakeTransport(_chat_response({"schema_version": "1.0"}))

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process("A clerk reviews an application.", transport=transport)

    assert captured.value.code == "schema_validation_failed"
    assert captured.value.details["issues"]


def test_invalid_graph_references_are_rejected_without_repair() -> None:
    transport = FakeTransport(_chat_response(_fixture("invalid_dangling_sequence_flow.json")))

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process("A clerk reviews an application.", transport=transport)

    assert captured.value.code == "graph_validation_failed"
    assert any(
        issue["code"] == "unknown_sequence_target"
        for issue in captured.value.details["issues"]
    )


@pytest.mark.parametrize(
    ("network_error", "code"),
    [
        (url_error.URLError("connection refused"), "ollama_unavailable"),
        (socket.timeout("timed out"), "ollama_timeout"),
    ],
)
def test_local_transport_failures_are_categorized(
    network_error: Exception, code: str
) -> None:
    with patch("app.llm.ollama.url_request.urlopen", side_effect=network_error) as call:
        with pytest.raises(OllamaExtractionError) as captured:
            extract_process("A clerk reviews an application.")

    assert captured.value.code == code
    assert call.call_count == 1


def test_unsupported_model_refusal_is_not_guessed_or_retried() -> None:
    transport = FakeTransport(
        _chat_response(
            {"error": "unsupported_description", "message": "Actors are unclear"}
        )
    )

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process("Something happens somehow.", transport=transport)

    assert captured.value.code == "unsupported_description"
    assert len(transport.calls) == 1


def test_empty_or_oversized_input_is_rejected_before_http() -> None:
    transport = FakeTransport(_chat_response(_fixture()))
    settings = OllamaSettings(max_input_chars=10)

    with pytest.raises(OllamaExtractionError) as empty_error:
        extract_process("   ", settings=settings, transport=transport)
    with pytest.raises(OllamaExtractionError) as size_error:
        extract_process("A process description", settings=settings, transport=transport)

    assert empty_error.value.code == "unsupported_description"
    assert size_error.value.code == "input_too_large"
    assert transport.calls == []


def test_environment_can_replace_model_and_local_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:22434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:14b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "12.5")
    transport = FakeTransport(_chat_response(_fixture()))

    extract_process("A clerk reviews an application.", transport=transport)

    url, payload, timeout, _ = transport.calls[0]
    assert url == "http://localhost:22434/api/chat"
    assert payload["model"] == "qwen3:14b"
    assert timeout == 12.5


def test_adapter_never_requests_model_download() -> None:
    transport = FakeTransport(_chat_response(_fixture()))

    extract_process("A clerk reviews an application.", transport=transport)

    assert len(transport.calls) == 1
    url, payload, _, _ = transport.calls[0]
    assert url.endswith("/api/chat")
    assert "/api/pull" not in url
    assert "pull" not in payload


def test_expected_api_routes_are_registered() -> None:
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/v1/generate-from-json" in paths
    assert "/api/v1/generate-from-text" in paths
    assert "/api/v1/health" in paths
