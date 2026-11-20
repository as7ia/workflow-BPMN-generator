"""Conservative deterministic normalization before/after one model repair."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph

REFERENCE_FIELDS = {
    "participant_id",
    "parent_scope_id",
    "subprocess_node_id",
    "lane_id",
    "scope_id",
    "default_flow_id",
    "source_ref",
    "target_ref",
}


def _at_path(document: Any, path: tuple[Any, ...]) -> Any:
    current = document
    for part in path:
        current = current[part]
    return current


def _set_path(document: Any, path: tuple[Any, ...], value: Any) -> None:
    parent = _at_path(document, path[:-1])
    parent[path[-1]] = value


def _delete_path(document: Any, path: tuple[Any, ...]) -> None:
    parent = _at_path(document, path[:-1])
    del parent[path[-1]]


def _collection_ids(candidate: dict[str, Any], *collections: str) -> set[str]:
    identifiers: set[str] = set()
    for collection in collections:
        values = candidate.get(collection)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                identifiers.add(value["id"])
    return identifiers


def _allowed_reference_ids(
    candidate: dict[str, Any], path: tuple[Any, ...]
) -> set[str]:
    field = path[-1]
    if field == "participant_id":
        return _collection_ids(candidate, "participants")
    if field in {"parent_scope_id", "scope_id"}:
        return _collection_ids(candidate, "scopes")
    if field == "subprocess_node_id":
        return {
            node["id"]
            for node in candidate.get("nodes", [])
            if isinstance(node, dict)
            and node.get("type") == "subprocess"
            and isinstance(node.get("id"), str)
        }
    if field == "lane_id":
        return _collection_ids(candidate, "lanes")
    if field == "default_flow_id":
        return _collection_ids(candidate, "sequence_flows")
    if field in {"source_ref", "target_ref"}:
        if path[0] == "sequence_flows":
            return _collection_ids(candidate, "nodes")
        if path[0] == "message_flows":
            return _collection_ids(candidate, "nodes", "participants")
    return set()


def apply_safe_deterministic_repairs(candidate: Any) -> tuple[Any, dict[str, Any]]:
    """Apply only information-preserving, uniquely resolvable corrections."""
    repaired = copy.deepcopy(candidate)
    report: dict[str, Any] = {
        "attempted": False,
        "changes": [],
        "refusals": [],
    }
    if not isinstance(repaired, dict):
        report["refusals"].append({"code": "candidate_not_object", "location": []})
        return repaired, report

    # Repeat because removing a harmless extra can expose another schema issue.
    for _ in range(4):
        try:
            model = ProcessModel.model_validate(repaired)
        except ValidationError as error:
            changed = False
            for issue in error.errors():
                path = tuple(issue["loc"])
                issue_type = issue["type"]
                try:
                    value = _at_path(repaired, path)
                except (KeyError, IndexError, TypeError):
                    continue
                if issue_type == "extra_forbidden":
                    if value is None:
                        _delete_path(repaired, path)
                        report["changes"].append({
                            "code": "remove_null_forbidden_field",
                            "location": list(path),
                        })
                        changed = True
                    else:
                        report["refusals"].append({
                            "code": "non_null_forbidden_field_requires_semantic_review",
                            "location": list(path),
                        })
                elif issue_type == "string_type" and path and path[-1] in REFERENCE_FIELDS:
                    if (
                        isinstance(value, dict)
                        and set(value) == {"id"}
                        and isinstance(value["id"], str)
                        and value["id"] in _allowed_reference_ids(repaired, path)
                    ):
                        _set_path(repaired, path, value["id"])
                        report["changes"].append({
                            "code": "unwrap_unique_existing_reference",
                            "location": list(path),
                            "resolved_id": value["id"],
                        })
                        changed = True
                    else:
                        report["refusals"].append({
                            "code": "reference_not_uniquely_resolvable",
                            "location": list(path),
                        })
            if not changed:
                break
            report["attempted"] = True
            continue
        else:
            try:
                validate_process_graph(model)
            except ProcessGraphValidationError as error:
                changed = False
                for issue in error.issues:
                    if issue.code != "invalid_gateway_default_flow":
                        continue
                    node_index = int(issue.location[1])
                    node = model.nodes[node_index]
                    outgoing = [
                        flow for flow in model.sequence_flows
                        if flow.source_ref == node.id
                    ]
                    unconditional = [flow for flow in outgoing if flow.condition is None]
                    if len(unconditional) != 1:
                        report["refusals"].append({
                            "code": "ambiguous_gateway_default_flow",
                            "location": list(issue.location),
                            "candidate_count": len(unconditional),
                        })
                        continue
                    selected = unconditional[0]
                    repaired["nodes"][node_index]["default_flow_id"] = selected.id
                    report["changes"].append({
                        "code": "resolve_unique_gateway_default_flow",
                        "location": list(issue.location),
                        "resolved_id": selected.id,
                    })
                    changed = True
                if changed:
                    report["attempted"] = True
                    continue
            break
    return repaired, report
