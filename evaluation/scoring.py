"""ID-independent semantic scoring for ProcessModel evaluation cases."""

from __future__ import annotations

from statistics import mean
from typing import Any

from app.domain import ProcessModel
from app.domain.process_model import NodeType


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _set_score(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    precision = len(expected & actual) / len(actual)
    recall = len(expected & actual) / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _activity_names(model: ProcessModel) -> set[str]:
    return {
        _norm(node.name)
        for node in model.nodes
        if node.type in {NodeType.TASK, NodeType.SUBPROCESS}
    }


def _gateway_score(model: ProcessModel, expected: dict[str, int]) -> float:
    actual = {
        node_type.value: sum(node.type == node_type for node in model.nodes)
        for node_type in (NodeType.EXCLUSIVE_GATEWAY, NodeType.PARALLEL_GATEWAY)
    }
    return mean(
        1.0 if actual.get(node_type, 0) == count else 0.0
        for node_type, count in expected.items()
    )


def _parallel_block_score(model: ProcessModel, blocks: list[dict[str, Any]]) -> float:
    if not blocks:
        return 1.0 if not any(node.type == NodeType.PARALLEL_GATEWAY for node in model.nodes) else 0.0
    nodes = {node.id: node for node in model.nodes}
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for flow in model.sequence_flows:
        outgoing.setdefault(flow.source_ref, set()).add(flow.target_ref)
        incoming.setdefault(flow.target_ref, set()).add(flow.source_ref)
    scores = []
    gateways = [node for node in model.nodes if node.type == NodeType.PARALLEL_GATEWAY]
    for block in blocks:
        branches = {_norm(name) for name in block["branches"]}
        split = any(
            branches <= {_norm(nodes[target].name) for target in outgoing.get(gateway.id, set()) if target in nodes}
            for gateway in gateways
        )
        joined = any(
            branches <= {_norm(nodes[source].name) for source in incoming.get(gateway.id, set()) if source in nodes}
            for gateway in gateways
        )
        scores.append((float(split) + float(joined or not block.get("join_required", True))) / 2)
    return mean(scores)


def _loop_score(model: ProcessModel, loops: list[dict[str, str]]) -> float:
    if not loops:
        return 1.0
    nodes = {node.id: node for node in model.nodes}
    actual = {
        (_norm(nodes[flow.source_ref].name), _norm(nodes[flow.target_ref].name))
        for flow in model.sequence_flows
        if flow.source_ref in nodes and flow.target_ref in nodes
    }
    expected = {(_norm(item["from"]), _norm(item["to"])) for item in loops}
    return len(expected & actual) / len(expected)


def _message_score(model: ProcessModel, messages: list[dict[str, str]]) -> float:
    if not messages:
        return 1.0 if not model.message_flows else 0.0
    nodes = {node.id: node for node in model.nodes}
    participants = {participant.id: participant for participant in model.participants}

    def endpoint(reference: str) -> tuple[str, str]:
        if reference in nodes:
            node = nodes[reference]
            return _norm(participants[node.participant_id].name), _norm(node.name)
        participant = participants[reference]
        return _norm(participant.name), ""

    actual = set()
    for flow in model.message_flows:
        source_participant, source_activity = endpoint(flow.source_ref)
        target_participant, target_activity = endpoint(flow.target_ref)
        actual.add(
            (
                source_participant,
                source_activity,
                target_participant,
                target_activity,
                _norm(flow.label or ""),
            )
        )
    expected = {
        (
            _norm(item["source_participant"]),
            _norm(item["source_activity"]),
            _norm(item["target_participant"]),
            _norm(item["target_activity"]),
            _norm(item.get("label", "")),
        )
        for item in messages
    }
    return _set_score(expected, actual)


def _subprocess_score(model: ProcessModel, expected: list[dict[str, Any]]) -> float:
    if not expected:
        return 1.0 if not any(node.type == NodeType.SUBPROCESS for node in model.nodes) else 0.0
    scopes_by_owner = {
        scope.subprocess_node_id: scope.id
        for scope in model.scopes
        if scope.subprocess_node_id is not None
    }
    scores = []
    for item in expected:
        owners = [
            node
            for node in model.nodes
            if node.type == NodeType.SUBPROCESS and _norm(node.name) == _norm(item["name"])
        ]
        if not owners:
            scores.append(0.0)
            continue
        child_scope = scopes_by_owner.get(owners[0].id)
        contained = {
            _norm(node.name)
            for node in model.nodes
            if node.scope_id == child_scope and node.type in {NodeType.TASK, NodeType.SUBPROCESS}
        }
        required = {_norm(name) for name in item["contains"]}
        scores.append(len(required & contained) / len(required) if required else 1.0)
    return mean(scores)


def score_candidate(model: ProcessModel, expected: dict[str, Any]) -> dict[str, Any]:
    """Score semantic fidelity without relying on exact generated identifiers."""
    expected_activities = {_norm(name) for name in expected["activities"]}
    actual_activities = _activity_names(model)
    expected_participants = {_norm(name) for name in expected["participants"]}
    actual_participants = {_norm(item.name) for item in model.participants}
    expected_lanes = {_norm(name) for name in expected.get("lanes", [])}
    actual_lanes = {_norm(item.name) for item in model.lanes}
    expected_conditions = {_norm(name) for name in expected.get("exclusive_conditions", [])}
    actual_conditions = {
        _norm(flow.condition)
        for flow in model.sequence_flows
        if flow.condition is not None
    }
    category_scores = {
        "required_activities": len(expected_activities & actual_activities) / len(expected_activities),
        "participants_and_lanes": mean(
            [_set_score(expected_participants, actual_participants), _set_score(expected_lanes, actual_lanes)]
        ),
        "gateway_types_and_branches": mean(
            [
                _gateway_score(model, expected["gateway_counts"]),
                _set_score(expected_conditions, actual_conditions),
            ]
        ),
        "parallel_split_join": _parallel_block_score(model, expected["parallel_blocks"]),
        "loop_preservation": _loop_score(model, expected["loops"]),
        "message_flow_direction": _message_score(model, expected["message_flows"]),
        "subprocess_nesting": _subprocess_score(model, expected["subprocesses"]),
        "invented_or_missing_elements": _set_score(expected_activities, actual_activities),
    }
    return {
        "overall": mean(category_scores.values()),
        "categories": category_scores,
        "missing_activities": sorted(expected_activities - actual_activities),
        "invented_activities": sorted(actual_activities - expected_activities),
    }
