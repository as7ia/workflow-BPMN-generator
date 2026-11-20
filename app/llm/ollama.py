"""Bounded local Ollama adapter for candidate ProcessModel extraction."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)


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
    think: bool | None = None  # None preserves Ollama default; False disables reasoning
    repair_invalid_graph: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OLLAMA_BASE_URL must not contain credentials, query, or fragment")
        if not self.model.strip():
            raise ValueError("OLLAMA_MODEL must not be empty")
        if self.think is not None and not isinstance(self.think, bool):
            raise ValueError("OLLAMA_THINK must be true or false")
        if not isinstance(self.repair_invalid_graph, bool):
            raise ValueError("OLLAMA_REPAIR_INVALID_GRAPH must be true or false")
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
        raw_think = os.getenv("OLLAMA_THINK")
        if raw_think is not None and raw_think.lower() not in {"true", "false", "1", "0"}:
            raise ValueError("OLLAMA_THINK must be true or false")
        think = None if raw_think is None else raw_think.lower() in {"true", "1"}
        raw_repair = os.getenv("OLLAMA_REPAIR_INVALID_GRAPH", "false")
        if raw_repair.lower() not in {"true", "false", "1", "0"}:
            raise ValueError("OLLAMA_REPAIR_INVALID_GRAPH must be true or false")
        return cls(
            think=think,
            repair_invalid_graph=raw_repair.lower() in {"true", "1"},
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

SUPPORTED ELEMENTS:
Use only start_event, end_event, task, subprocess, exclusive_gateway, and parallel_gateway.
Preserve explicitly stated actors, responsibilities, tasks, ordering, decisions, parallel activities, loops, communications, and explicitly bounded subprocesses.
Do not invent business facts.

SCOPES:
Every participant must have exactly one root scope.
For simple linear processes, use only the root scope. Do not create subprocesses.
Create a subprocess only when the description explicitly defines a bounded nested activity.
Every scope must contain a start event and an end event.
Every nested scope must reference an existing node of type subprocess through subprocess_node_id.
Every subprocess node must correspond to exactly one nested scope.
Do not create orphan scopes or empty nested scopes.

SEQUENCE FLOWS:
Every sequence flow source_ref and target_ref must reference existing nodes.
Source and target must belong to the same scope.
Do not connect nodes from different scopes using sequence flows.
Every non-start node must be reachable from a start event in its scope.
Every non-end node must be able to reach an end event in its scope.
For a linear process, connect start -> tasks in stated order -> end.
Preserve explicitly described loops and branches without inventing additional activities.

BPMN SEMANTICS:
Use exclusive gateways for decisions and parallel gateways for simultaneous branches.
Conditions must be short human-readable English labels.
Message flows connect different participants and may only reference tasks or subprocesses.
Keep sequence flows and message flows distinct.

OUTPUT:
Return exactly one JSON object matching the supplied ProcessModel schema.
Before responding, check every node reference, scope reference, sequence flow, and required start/end event.
If the description cannot be represented without guessing, return {"error":"unsupported_description","message":"brief reason"}.
All IDs must be globally unique across participants, lanes, scopes,
nodes, sequence flows, and message flows.
Never reuse a scope ID as a participant ID, node ID, or flow ID.
Use distinct prefixes such as participant_, scope_, node_, flow_.
Identifiers must start with an ASCII letter and contain only letters, digits, underscore, dot, or hyphen.
schema_version must be "1.0".
"""


def _request_payload(description: str, settings: OllamaSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "stream": False,
        **({"think": settings.think} if settings.think is not None else {}),
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


def _repair_payload(
    description: str,
    candidate: Any,
    issues: list[dict[str, Any]],
    settings: OllamaSettings,
    *,
    validation_stage: str = "graph",
) -> dict[str, Any]:
    if validation_stage not in {"schema", "graph"}:
        raise ValueError("validation_stage must be schema or graph")
    repair_request = {
        "original_description": description,
        "original_candidate": candidate,
        **(
            {"graph_validation_issues": issues}
            if validation_stage == "graph"
            else {"schema_validation_issues": issues}
        ),
    }
    prompt = """Repair the supplied ProcessModel JSON using the schema or graph diagnostics and rules below.
Return exactly one JSON object matching ProcessModel schema 1.0, with no explanation.

PRESERVE BUSINESS MEANING:
Treat original_description as the source of truth for business meaning. Use the candidate only
as the structure to repair and the diagnostics only as structural defects to correct.
Preserve every business activity, participant, decision, branch, loop, condition, ordering,
responsibility, and message. Make only structural corrections required by the diagnostics.
Never invent, remove, merge, or rename business activities or participants. Never create a
subprocess node merely to fix a scope error. A missing nested scope may be added only when the
candidate already contains a subprocess node that requires its internal scope. Do not turn an
ordinary task or group of tasks into a subprocess without explicit evidence in
original_description. If the candidate contains a subprocess that is not justified by
original_description, do not preserve that invented subprocess: keep its business activities
and ordering, but represent them without the unjustified subprocess boundary.

ROOT SCOPE VERSUS NESTED SCOPE:
- Each participant must own exactly one root scope.
- A root scope has that participant_id and BOTH parent_scope_id=null and
  subprocess_node_id=null. It is the participant's top-level process, not subprocess contents.
- A nested scope represents the contents inside one existing subprocess node. It must set BOTH
  parent_scope_id and subprocess_node_id; never set only one of them.
- parent_scope_id must identify the existing scope that directly contains the subprocess node.
- subprocess_node_id must identify that existing node, whose type must be "subprocess".
- The subprocess node's scope_id must equal the nested scope's parent_scope_id.
- The nested scope, its subprocess node, and its parent scope must have the same participant_id.
- Each subprocess node owns exactly one nested scope, and each nested scope belongs to exactly
  one subprocess node. Do not create orphan or duplicate nested scopes.

SUBPROCESS BOUNDARY:
Treat the parent scope and the nested scope as two separate control-flow graphs. The subprocess
node itself stays in its parent scope. It must be reachable from a start event in that parent
scope and must be able to reach an end event in that parent scope, using only parent-scope nodes
and sequence flows. In other words, the parent graph enters and leaves the subprocess through
the subprocess node; it never connects directly to an internal activity or internal event.

The subprocess's internal start event, internal activities, gateways, and internal end event
all belong to its nested scope. Connect those internal nodes only to other nodes in that same
nested scope. A sequence flow's source_ref and target_ref must always resolve to nodes with the
same scope_id. Never create a sequence flow from a parent-scope node to a nested-scope node, or
from a nested-scope node to a parent-scope node.

PER-SCOPE CONNECTIVITY:
Independently inspect every root and nested scope. Each scope must contain at least one
start_event and at least one end_event whose scope_id and participant_id match that scope.
Within each scope, every node must be reachable by same-scope sequence flows from a start event,
and every node must be able to reach an end event. A start event has no incoming sequence flow;
an end event has no outgoing sequence flow. Preserve existing branches, joins, conditions, and
loops.

SCHEMA EXACTNESS AND REFERENCES:
- Emit only fields declared by the supplied ProcessModel schema. Never emit parent_lane_id.
- A Lane object has only id, name, and participant_id. lane_id is permitted on a process node,
  not on a Lane, Participant, ProcessScope, SequenceFlow, or MessageFlow object.
- Every participant_id, lane_id, scope_id, parent_scope_id, subprocess_node_id,
  default_flow_id, source_ref, and target_ref value must be either a JSON string or null when
  the schema permits null. Never emit an object, array, or number in a reference field.
- Copy an existing ID string exactly. Change a reference only when exactly one existing element
  of the required type is the valid target. If several targets are possible, do not guess.

EXCLUSIVE GATEWAY DEFAULTS:
An exclusive gateway's default_flow_id must equal the id of one of that gateway's existing
outgoing sequence flows. That selected default flow must have condition=null; every other
outgoing branch must keep its explicit condition. If exactly one existing outgoing flow is
unconditional, it is the only safe default. Never add, remove, merge, or retarget a business
branch merely to repair default_flow_id.

You may add a purely technical start_event or end_event to a scope that is missing one only when
doing so does not add or change business meaning. Give it a neutral technical name. Connect a
new start event only to an unambiguous existing entry node in the same scope, and connect an
unambiguous existing exit node in that scope only to a new end event. These technical events are
not business activities. Do not use them to invent an activity, branch, condition, loop, or
ordering. If the entry or exit is ambiguous, do not guess.

Add or redirect a sequence flow only when the candidate's existing ordering, branch condition,
loop, or subprocess boundary makes the intended connection unambiguous. Never guess a business
connection, branch, condition, or ordering merely to make the graph connected. If a valid repair
would require an ambiguous business connection, do not invent one; the result will be rejected.

FINAL CHECK:
First correct every supplied schema diagnostic without adding undeclared fields. Then recheck
the complete schema, participant root scopes, nested-scope pairs, subprocess ownership,
same-scope sequence flows, and reachability in every scope before responding.
"""
    return {
        "model": settings.model,
        "stream": False,
        **({"think": settings.think} if settings.think is not None else {}),
        "format": ProcessModel.model_json_schema(),
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(repair_request, separators=(",", ":")),
            },
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


def _parse_candidate(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise OllamaExtractionError(
            "malformed_output", "Ollama message content is not valid JSON."
        ) from error


def _validate_candidate(candidate: Any) -> ProcessModel:
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
    candidate = _parse_candidate(content)
    if isinstance(candidate, dict) and candidate.get("error") == "unsupported_description":
        raise OllamaExtractionError(
            "unsupported_description",
            "Ollama could not represent the description without guessing.",
        )
    initial_issues: list[dict[str, Any]]
    try:
        return _validate_candidate(candidate)
    except OllamaExtractionError as initial_error:
        if initial_error.code != "graph_validation_failed" or not active_settings.repair_invalid_graph:
            raise
        initial_issues = initial_error.details["issues"]

    logger.info(
        "Attempting ProcessModel graph repair; issue_codes=%s",
        [issue["code"] for issue in initial_issues],
    )
    try:
        repair_response = client.post_json(
            f"{active_settings.base_url}/api/chat",
            _repair_payload(normalized, candidate, initial_issues, active_settings),
            timeout_seconds=active_settings.timeout_seconds,
            max_response_bytes=active_settings.max_response_bytes,
        )
        repaired = _parse_candidate(_response_content(repair_response))
        model = _validate_candidate(repaired)
    except OllamaExtractionError as repair_error:
        logger.warning("ProcessModel graph repair failed; code=%s", repair_error.code)
        raise OllamaExtractionError(
            repair_error.code,
            str(repair_error),
            details={
                **repair_error.details,
                "repair_attempted": True,
                "initial_graph_issues": initial_issues,
            },
        ) from repair_error
    logger.info("ProcessModel graph repair succeeded")
    return model
