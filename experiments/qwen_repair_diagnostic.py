"""Explicit local diagnostic comparing Qwen candidates before and after repair."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph
from app.llm import OllamaExtractionError, OllamaSettings
from app.llm import ollama

_COLLECTIONS = ("participants", "scopes", "nodes", "sequence_flows")
_STRUCTURAL_CUES = (
    "subprocess",
    "sub-process",
    "nested process",
    "inside the process",
    "consists of the following steps",
    "comprises the following steps",
)
_NAME_STOPWORDS = {
    "a",
    "an",
    "and",
    "of",
    "process",
    "subprocess",
    "the",
    "to",
}


@dataclass(frozen=True, slots=True)
class DiagnosticCapture:
    """In-memory candidates and their safe topological comparison report."""

    initial_candidate: dict[str, Any]
    repaired_candidate: dict[str, Any]
    report: dict[str, Any]


def _validation_issues(model: ProcessModel) -> list[dict[str, Any]]:
    try:
        validate_process_graph(model)
    except ProcessGraphValidationError as error:
        return [
            {
                "code": issue.code,
                "location": list(issue.location),
            }
            for issue in error.issues
        ]
    return []


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        field: {"before": before.get(field), "after": after.get(field)}
        for field in sorted(before.keys() | after.keys())
        if before.get(field) != after.get(field)
    }


def _compare_collection(
    before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]
) -> dict[str, Any]:
    before = {item["id"]: item for item in before_items}
    after = {item["id"]: item for item in after_items}
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": {
            item_id: _changed_fields(before[item_id], after[item_id])
            for item_id in sorted(before.keys() & after.keys())
            if before[item_id] != after[item_id]
        },
    }


def _item_at(candidate: dict[str, Any], collection: str, index: int) -> dict[str, Any] | None:
    items = candidate.get(collection)
    if not isinstance(items, list) or index >= len(items):
        return None
    item = items[index]
    return item if isinstance(item, dict) else None


def _node_focus(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    node = _item_at(candidate, "nodes", index)
    if node is None:
        return {"item": None, "incoming_flow_ids": [], "outgoing_flow_ids": []}
    node_id = node["id"]
    flows = candidate["sequence_flows"]
    return {
        "item": node,
        "incoming_flow_ids": sorted(
            flow["id"] for flow in flows if flow["target_ref"] == node_id
        ),
        "outgoing_flow_ids": sorted(
            flow["id"] for flow in flows if flow["source_ref"] == node_id
        ),
    }


def _scope_focus(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    scope = _item_at(candidate, "scopes", index)
    if scope is None:
        return {
            "item": None,
            "node_ids": [],
            "start_event_ids": [],
            "end_event_ids": [],
            "internal_flow_ids": [],
        }
    nodes = [node for node in candidate["nodes"] if node["scope_id"] == scope["id"]]
    node_ids = {node["id"] for node in nodes}
    return {
        "item": scope,
        "node_ids": sorted(node_ids),
        "start_event_ids": sorted(
            node["id"] for node in nodes if node["type"] == "start_event"
        ),
        "end_event_ids": sorted(
            node["id"] for node in nodes if node["type"] == "end_event"
        ),
        "internal_flow_ids": sorted(
            flow["id"]
            for flow in candidate["sequence_flows"]
            if flow["source_ref"] in node_ids and flow["target_ref"] in node_ids
        ),
    }


def _normalized_words(value: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", value.lower())
        if len(word) > 2 and word not in _NAME_STOPWORDS
    }


def _subprocess_justification(
    description: str,
    initial: dict[str, Any],
    repaired: dict[str, Any],
) -> list[dict[str, Any]]:
    initial_nodes = {
        node["id"]: node
        for node in initial["nodes"]
        if node.get("type") == "subprocess"
    }
    repaired_nodes = {
        node["id"]: node
        for node in repaired["nodes"]
        if node.get("type") == "subprocess"
    }
    lowered_description = description.lower()
    has_structural_cue = any(cue in lowered_description for cue in _STRUCTURAL_CUES)
    description_words = _normalized_words(description)
    findings: list[dict[str, Any]] = []
    for node_id, node in sorted(repaired_nodes.items()):
        name_words = _normalized_words(str(node.get("name", "")))
        matching_name_words = sorted(name_words & description_words)
        added_by_repair = node_id not in initial_nodes
        if added_by_repair and has_structural_cue and matching_name_words:
            status = "added_by_repair_with_description_evidence"
            reason = (
                "The repair added the subprocess, but the description has a nesting cue "
                "and words matching its name; human review is still required."
            )
        elif added_by_repair:
            status = "unjustified_added_by_repair"
            reason = "The repair introduced a subprocess without explicit text evidence."
        elif has_structural_cue and matching_name_words:
            status = "explicit_description_evidence"
            reason = "The description has a nesting cue and words matching the subprocess name."
        else:
            status = "not_demonstrated"
            reason = "No explicit nesting cue plus subprocess-name match was found."
        findings.append(
            {
                "id": node_id,
                "name": node.get("name"),
                "added_by_repair": added_by_repair,
                "status": status,
                "reason": reason,
                "matching_name_words": matching_name_words,
            }
        )
    return findings


def compare_candidates(
    description: str,
    initial_model: ProcessModel,
    repaired_model: ProcessModel,
) -> dict[str, Any]:
    """Build an ID-based topological diff without exposing either full candidate."""
    initial = initial_model.model_dump(mode="json")
    repaired = repaired_model.model_dump(mode="json")
    return {
        "initial_graph_issues": _validation_issues(initial_model),
        "repaired_graph_issues": _validation_issues(repaired_model),
        "collections": {
            name: _compare_collection(initial[name], repaired[name])
            for name in _COLLECTIONS
        },
        "focus": {
            "nodes[5]": {
                "before": _node_focus(initial, 5),
                "after": _node_focus(repaired, 5),
            },
            "scopes[1]": {
                "before": _scope_focus(initial, 1),
                "after": _scope_focus(repaired, 1),
            },
        },
        "subprocess_justification": _subprocess_justification(
            description, initial, repaired
        ),
        "justification_limit": (
            "This is a conservative text heuristic, not proof of business intent. "
            "A human must review every not_demonstrated finding."
        ),
    }


def capture_live_diagnostic(
    description: str,
    *,
    settings: OllamaSettings,
    transport: ollama.OllamaTransport | None = None,
) -> DiagnosticCapture:
    """Make exactly one initial and one repair call and retain candidates in memory."""
    normalized = description.strip()
    if not normalized:
        raise ValueError("The diagnostic description must not be empty")
    if len(normalized) > settings.max_input_chars:
        raise ValueError("The diagnostic description exceeds max_input_chars")

    client = transport or ollama._UrllibTransport()
    initial_response = client.post_json(
        f"{settings.base_url}/api/chat",
        ollama._request_payload(normalized, settings),
        timeout_seconds=settings.timeout_seconds,
        max_response_bytes=settings.max_response_bytes,
    )
    initial_candidate = ollama._parse_candidate(
        ollama._response_content(initial_response)
    )
    initial_model = ProcessModel.model_validate(initial_candidate)
    initial_issues = _validation_issues(initial_model)
    if not initial_issues:
        raise ValueError("The initial candidate is valid; there is nothing to repair")

    repaired_response = client.post_json(
        f"{settings.base_url}/api/chat",
        ollama._repair_payload(initial_candidate, initial_issues, settings),
        timeout_seconds=settings.timeout_seconds,
        max_response_bytes=settings.max_response_bytes,
    )
    repaired_candidate = ollama._parse_candidate(
        ollama._response_content(repaired_response)
    )
    repaired_model = ProcessModel.model_validate(repaired_candidate)
    report = compare_candidates(normalized, initial_model, repaired_model)
    return DiagnosticCapture(initial_candidate, repaired_candidate, report)


def render_report(report: dict[str, Any]) -> str:
    """Render the topological comparison; full candidates are intentionally omitted."""
    lines = ["Qwen ProcessModel repair topology diagnostic", ""]
    lines.append(f"Initial issues: {json.dumps(report['initial_graph_issues'])}")
    lines.append(f"Repaired issues: {json.dumps(report['repaired_graph_issues'])}")
    for name, changes in report["collections"].items():
        lines.extend(
            [
                "",
                name.upper(),
                f"  added: {changes['added']}",
                f"  removed: {changes['removed']}",
                f"  changed: {json.dumps(changes['changed'], ensure_ascii=False)}",
            ]
        )
    lines.extend(["", "FOCUS"])
    for location, values in report["focus"].items():
        lines.append(f"  {location}: {json.dumps(values, ensure_ascii=False)}")
    lines.extend(["", "SUBPROCESS JUSTIFICATION"])
    for finding in report["subprocess_justification"]:
        lines.append(
            f"  {finding['id']}: {finding['status']} - {finding['reason']}"
        )
    lines.append(f"  Limit: {report['justification_limit']}")
    return "\n".join(lines)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit local comparison of Qwen candidates before and after repair"
    )
    parser.add_argument("--confirm-live-ollama", action="store_true")
    parser.add_argument("--description-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.confirm_live_ollama:
        raise SystemExit("Refusing inference: pass --confirm-live-ollama explicitly")
    description = args.description_file.read_text(encoding="utf-8")
    try:
        capture = capture_live_diagnostic(
            description,
            settings=OllamaSettings.from_env(),
        )
    except (OllamaExtractionError, ValueError) as error:
        raise SystemExit(f"Diagnostic failed: {type(error).__name__}") from error
    print(render_report(capture.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
