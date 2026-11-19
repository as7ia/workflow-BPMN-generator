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
    def __init__(self, response: dict[str, object] | list[dict[str, object]]) -> None:
        self.responses = response if isinstance(response, list) else [response]
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
        return self.responses[len(self.calls) - 1]


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
    assert len(transport.calls) == 1


def test_opt_in_graph_repair_sends_diagnostics_and_returns_revalidated_model() -> None:
    invalid = _fixture("invalid_dangling_sequence_flow.json")
    repaired = _fixture()
    transport = FakeTransport([_chat_response(invalid), _chat_response(repaired)])

    model = extract_process(
        "A clerk reviews an application.",
        settings=OllamaSettings(repair_invalid_graph=True),
        transport=transport,
    )

    assert model.id == "linear"
    assert len(transport.calls) == 2
    repair_payload = transport.calls[1][1]
    repair_input = json.loads(repair_payload["messages"][1]["content"])
    assert repair_input["original_candidate"] == invalid
    assert any(
        issue["code"] == "unknown_sequence_target"
        for issue in repair_input["graph_validation_issues"]
    )
    assert "Do not invent" in repair_payload["messages"][0]["content"]


def test_failed_graph_repair_is_refused_after_exactly_two_calls() -> None:
    invalid = _fixture("invalid_dangling_sequence_flow.json")
    transport = FakeTransport([_chat_response(invalid), _chat_response(invalid)])

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process(
            "A clerk reviews an application.",
            settings=OllamaSettings(repair_invalid_graph=True),
            transport=transport,
        )

    assert captured.value.code == "graph_validation_failed"
    assert captured.value.details["repair_attempted"] is True
    assert captured.value.details["initial_graph_issues"]
    assert len(transport.calls) == 2


def test_repair_is_not_attempted_for_schema_failure() -> None:
    transport = FakeTransport(_chat_response({"schema_version": "1.0"}))

    with pytest.raises(OllamaExtractionError) as captured:
        extract_process(
            "A clerk reviews an application.",
            settings=OllamaSettings(repair_invalid_graph=True),
            transport=transport,
        )

    assert captured.value.code == "schema_validation_failed"
    assert len(transport.calls) == 1


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


def test_opt_in_no_thinking_is_passed_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_THINK", "false")
    transport = FakeTransport(_chat_response(_fixture()))
    extract_process("A clerk reviews an application.", transport=transport)
    assert transport.calls[0][1]["think"] is False


def test_invalid_thinking_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_THINK", "maybe")
    with pytest.raises(ValueError, match="OLLAMA_THINK"):
        OllamaSettings.from_env()


def test_graph_repair_requires_explicit_valid_environment_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert OllamaSettings.from_env().repair_invalid_graph is False
    monkeypatch.setenv("OLLAMA_REPAIR_INVALID_GRAPH", "true")
    assert OllamaSettings.from_env().repair_invalid_graph is True
    monkeypatch.setenv("OLLAMA_REPAIR_INVALID_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="OLLAMA_REPAIR_INVALID_GRAPH"):
        OllamaSettings.from_env()
