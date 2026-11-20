"""Mock-only tests for bounded local Ollama process extraction."""

import json
import socket
from pathlib import Path
from unittest.mock import patch
from urllib import error as url_error

import pytest

from app.domain import ProcessModel
from app.llm import OllamaExtractionError, OllamaSettings, extract_process
from app.llm import ollama

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
    assert repair_input["original_description"] == "A clerk reviews an application."
    assert repair_input["original_candidate"] == invalid
    assert any(
        issue["code"] == "unknown_sequence_target"
        for issue in repair_input["graph_validation_issues"]
    )
    assert "Never invent" in repair_payload["messages"][0]["content"]


def test_repair_prompt_encodes_exact_root_and_nested_scope_invariants() -> None:
    issues = [
        {"code": "missing_participant_root_scope", "location": ["participants", 0, "id"]},
        {"code": "invalid_nested_scope", "location": ["scopes", 0]},
        {"code": "invalid_subprocess_parent", "location": ["scopes", 1, "parent_scope_id"]},
    ]

    description = "A manager reviews a request in a bounded subprocess."
    payload = ollama._repair_payload(
        description, _fixture(), issues, OllamaSettings()
    )
    prompt = " ".join(payload["messages"][0]["content"].split())

    assert "BOTH parent_scope_id=null and subprocess_node_id=null" in prompt
    assert "must set BOTH" in prompt
    assert "subprocess node's scope_id must equal the nested scope's parent_scope_id" in prompt
    assert "must have the same participant_id" in prompt
    assert "Each subprocess node owns exactly one nested scope" in prompt
    assert "original_description as the source of truth" in prompt
    assert "do not preserve that invented subprocess" in prompt
    request = json.loads(payload["messages"][1]["content"])
    assert request["original_description"] == description
    assert request["graph_validation_issues"] == issues


def test_repair_payload_accepts_schema_diagnostics_without_mislabeling_them() -> None:
    issues = [{"location": ["nodes", 0, "lane_id"], "type": "extra_forbidden"}]

    payload = ollama._repair_payload(
        "A clerk reviews an application.",
        {"schema_version": "1.0"},
        issues,
        OllamaSettings(),
        validation_stage="schema",
    )
    request = json.loads(payload["messages"][1]["content"])

    assert request["schema_validation_issues"] == issues
    assert "graph_validation_issues" not in request


def test_repair_prompt_keeps_subprocess_boundaries_and_forbids_ambiguous_flows() -> None:
    issues = [
        {"code": "unreachable_from_start", "location": ["nodes", 2]},
        {"code": "cannot_reach_end", "location": ["nodes", 2]},
    ]

    payload = ollama._repair_payload(
        "A clerk reviews an application.", _fixture(), issues, OllamaSettings()
    )
    prompt = " ".join(payload["messages"][0]["content"].split())

    assert "parent scope and the nested scope as two separate control-flow graphs" in prompt
    assert "subprocess node itself stays in its parent scope" in prompt
    assert "reachable from a start event in that parent scope" in prompt
    assert "able to reach an end event in that parent scope" in prompt
    assert "parent graph enters and leaves the subprocess through the subprocess node" in prompt
    assert "source_ref and target_ref" in prompt and "same scope_id" in prompt
    assert "Never create a sequence flow from a parent-scope node" in prompt
    assert "every node must be reachable" in prompt
    assert "every node must be able to reach an end event" in prompt
    assert "only when" in prompt and "unambiguous" in prompt
    assert "Never guess a business" in prompt
    assert "Never create a subprocess node merely to fix a scope error" in prompt
    assert "Never emit parent_lane_id" in prompt
    assert "Never emit an object, array, or number in a reference field" in prompt
    assert "If several targets are possible, do not guess" in prompt
    assert "default_flow_id must equal the id" in prompt
    assert "exactly one existing outgoing flow is unconditional" in prompt


def test_repair_prompt_allows_only_semantically_neutral_missing_scope_events() -> None:
    issues = [
        {"code": "unreachable_from_start", "location": ["nodes", 5]},
        {"code": "cannot_reach_end", "location": ["nodes", 5]},
        {"code": "missing_start_event", "location": ["scopes", 1]},
        {"code": "missing_end_event", "location": ["scopes", 1]},
    ]

    payload = ollama._repair_payload(
        "A clerk reviews an application.", _fixture(), issues, OllamaSettings()
    )
    prompt = " ".join(payload["messages"][0]["content"].split())

    assert "Each scope must contain at least one start_event and at least one end_event" in prompt
    assert "scope_id and participant_id match that scope" in prompt
    assert "purely technical start_event or end_event" in prompt
    assert "only when doing so does not add or change business meaning" in prompt
    assert "unambiguous existing entry node in the same scope" in prompt
    assert "unambiguous existing exit node in that scope" in prompt
    assert "If the entry or exit is ambiguous, do not guess" in prompt
    assert "not business activities" in prompt
    request = json.loads(payload["messages"][1]["content"])
    assert request["graph_validation_issues"] == issues


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
