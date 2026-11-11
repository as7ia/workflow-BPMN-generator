"""Bounded local Ollama adapter for candidate ProcessModel extraction."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import urlparse

from pydantic import ValidationError

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen3:8b"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_INPUT_CHARS = 12_000
_DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_DEFAULT_NUM_PREDICT = 4_096
_DEFAULT_NUM_CTX = 8_192


class OllamaExtractionError(RuntimeError):
    """Categorized failure from local process extraction."""

    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Environment-configurable, bounded Ollama inference settings."""

    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES
    num_predict: int = _DEFAULT_NUM_PREDICT
    num_ctx: int = _DEFAULT_NUM_CTX

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OLLAMA_BASE_URL must not contain credentials, query, or fragment")
        if not self.model.strip():
            raise ValueError("OLLAMA_MODEL must not be empty")
        bounded_values = {
            "OLLAMA_TIMEOUT_SECONDS": self.timeout_seconds,
            "OLLAMA_MAX_INPUT_CHARS": self.max_input_chars,
            "OLLAMA_MAX_RESPONSE_BYTES": self.max_response_bytes,
            "OLLAMA_NUM_PREDICT": self.num_predict,
            "OLLAMA_NUM_CTX": self.num_ctx,
        }
        for name, value in bounded_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(cls) -> OllamaSettings:
        """Load settings from the process environment without network access."""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL),
            timeout_seconds=float(
                os.getenv("OLLAMA_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))
            ),
            max_input_chars=int(
                os.getenv("OLLAMA_MAX_INPUT_CHARS", str(_DEFAULT_MAX_INPUT_CHARS))
            ),
            max_response_bytes=int(
                os.getenv(
                    "OLLAMA_MAX_RESPONSE_BYTES", str(_DEFAULT_MAX_RESPONSE_BYTES)
                )
            ),
            num_predict=int(
                os.getenv("OLLAMA_NUM_PREDICT", str(_DEFAULT_NUM_PREDICT))
            ),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", str(_DEFAULT_NUM_CTX))),
        )


class OllamaTransport(Protocol):
    """Injectable JSON transport used by the adapter and mock tests."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> dict[str, Any]: ...


class _UrllibTransport:
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = url_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with url_request.urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read(max_response_bytes + 1)
        except url_error.HTTPError as error:
            if error.code == 404:
                raise OllamaExtractionError(
                    "model_unavailable", "The configured Ollama model is unavailable."
                ) from error
            raise OllamaExtractionError(
                "ollama_unavailable", "The configured Ollama service rejected the request."
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise OllamaExtractionError(
                "ollama_timeout", "Ollama did not respond before the timeout."
            ) from error
        except (url_error.URLError, OSError) as error:
            raise OllamaExtractionError(
                "ollama_unavailable", "The configured Ollama service is unavailable."
            ) from error
        if len(response_body) > max_response_bytes:
            raise OllamaExtractionError(
                "response_too_large",
                "Ollama response exceeded the configured byte limit.",
                details={"max_response_bytes": max_response_bytes},
            )
        try:
            decoded = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaExtractionError(
                "malformed_response", "Ollama returned a malformed JSON response."
            ) from error
        if not isinstance(decoded, dict):
            raise OllamaExtractionError(
                "malformed_response", "Ollama response must be a JSON object."
            )
        return decoded


def _system_prompt() -> str:
    return """You extract English business process descriptions into ProcessModel schema 1.0 JSON.
Return JSON only. Never generate BPMN XML, BPMN DI, executable code, or explanations.
Use only these node types: start_event, end_event, task, subprocess, exclusive_gateway, parallel_gateway.
Preserve explicitly stated actors as participants, responsibilities as lanes, ordered tasks as sequence flows, decisions as exclusive gateways, simultaneous branches as parallel gateways, correction/repetition as ordinary sequence-flow loops, communications between different participants as message flows, and explicitly bounded nested work as subprocess scopes.
Every participant needs exactly one root scope. Every scope needs a start and end event. Each subprocess needs exactly one nested scope. Conditions are short human-readable English branch labels, never executable expressions. Message-flow node endpoints may only be tasks or subprocesses.
Do not invent actors, decisions, conditions, ordering, messages, subprocesses, or missing business facts. If the text is ambiguous or cannot be represented by the supported subset, return {\"error\":\"unsupported_description\",\"message\":\"brief reason\"} instead of guessing.
Identifiers must start with an ASCII letter and contain only letters, digits, underscore, dot, or hyphen. The schema_version is always \"1.0\"."""


def _request_payload(description: str, settings: OllamaSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "stream": False,
        "format": ProcessModel.model_json_schema(),
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": description},
        ],
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": settings.num_predict,
            "num_ctx": settings.num_ctx,
        },
    }


def _response_content(response: dict[str, Any]) -> str:
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaExtractionError(
            "malformed_response", "Ollama response has no message content."
        )
    return content


def extract_process(
    description: str,
    *,
    settings: OllamaSettings | None = None,
    transport: OllamaTransport | None = None,
) -> ProcessModel:
    """Extract and validate one candidate process using one local Ollama call."""
    active_settings = settings or OllamaSettings.from_env()
    normalized = description.strip()
    if not normalized or not any(char.isascii() and char.isalpha() for char in normalized):
        raise OllamaExtractionError(
            "unsupported_description",
            "A non-empty English business process description is required.",
        )
    if len(normalized) > active_settings.max_input_chars:
        raise OllamaExtractionError(
            "input_too_large",
            "Business process description exceeds the configured character limit.",
            details={"max_input_chars": active_settings.max_input_chars},
        )

    client = transport or _UrllibTransport()
    response = client.post_json(
        f"{active_settings.base_url}/api/chat",
        _request_payload(normalized, active_settings),
        timeout_seconds=active_settings.timeout_seconds,
        max_response_bytes=active_settings.max_response_bytes,
    )
    content = _response_content(response)
    try:
        candidate = json.loads(content)
    except json.JSONDecodeError as error:
        raise OllamaExtractionError(
            "malformed_output", "Ollama message content is not valid JSON."
        ) from error
    if isinstance(candidate, dict) and candidate.get("error") == "unsupported_description":
        raise OllamaExtractionError(
            "unsupported_description",
            "Ollama could not represent the description without guessing.",
        )
    try:
        model = ProcessModel.model_validate(candidate)
    except ValidationError as error:
        details = [
            {"location": list(item["loc"]), "type": item["type"]}
            for item in error.errors()
        ]
        raise OllamaExtractionError(
            "schema_validation_failed",
            "Ollama output does not match ProcessModel schema 1.0.",
            details={"issues": details},
        ) from error
    try:
        validate_process_graph(model)
    except ProcessGraphValidationError as error:
        details = [
            {"code": issue.code, "location": list(issue.location)}
            for issue in error.issues
        ]
        raise OllamaExtractionError(
            "graph_validation_failed",
            "Ollama output contains an invalid process graph.",
            details={"issues": details},
        ) from error
    return model
