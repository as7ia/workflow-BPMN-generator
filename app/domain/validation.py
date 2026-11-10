"""Deterministic structural and graph validation for process models."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from app.domain.process_model import NodeType, ProcessModel, ProcessNode, SequenceFlow

Location = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable validation failure with a JSON-oriented location."""

    code: str
    location: Location
    message: str


class ProcessGraphValidationError(ValueError):
    """Raised when a parsed process model is structurally inconsistent."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        summary = "; ".join(
            f"{issue.code} at {'.'.join(map(str, issue.location))}: {issue.message}"
            for issue in self.issues
        )
        super().__init__(summary)


def _issue(code: str, location: Location, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, location=location, message=message)


def _reachable(seed_ids: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    """Traverse a directed graph without recursion and safely handle cycles."""
    visited: set[str] = set()
    pending = deque(seed_ids)
    while pending:
        node_id = pending.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(adjacency.get(node_id, set()) - visited)
    return visited


def _validate_unique_ids(model: ProcessModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen: dict[str, Location] = {model.id: ("id",)}
    collections = (
        ("participants", model.participants),
        ("lanes", model.lanes),
        ("scopes", model.scopes),
        ("nodes", model.nodes),
        ("sequence_flows", model.sequence_flows),
        ("message_flows", model.message_flows),
    )
    for collection_name, elements in collections:
        for index, element in enumerate(elements):
            location = (collection_name, index, "id")
            previous = seen.get(element.id)
            if previous is not None:
                issues.append(
                    _issue(
                        "duplicate_id",
                        location,
                        f"ID '{element.id}' is already used at {'.'.join(map(str, previous))}.",
                    )
                )
            else:
                seen[element.id] = location
    return issues


def _validate_scope_hierarchy(model: ProcessModel) -> list[ValidationIssue]:
    """Validate root ownership and the finite subprocess scope hierarchy."""
    issues: list[ValidationIssue] = []
    root_scopes_by_participant: dict[str, list[int]] = defaultdict(list)
    child_scopes_by_subprocess: dict[str, list[int]] = defaultdict(list)
    scope_index_by_id = {scope.id: index for index, scope in enumerate(model.scopes)}

    for scope_index, scope in enumerate(model.scopes):
        if scope.parent_scope_id is None and scope.subprocess_node_id is None:
            root_scopes_by_participant[scope.participant_id].append(scope_index)
        if scope.subprocess_node_id is not None:
            child_scopes_by_subprocess[scope.subprocess_node_id].append(scope_index)

    for participant_index, participant in enumerate(model.participants):
        root_indexes = root_scopes_by_participant.get(participant.id, [])
        if not root_indexes:
            issues.append(
                _issue(
                    "missing_participant_root_scope",
                    ("participants", participant_index, "id"),
                    f"Participant '{participant.id}' must own exactly one root process scope.",
                )
            )
        elif len(root_indexes) > 1:
            issues.append(
                _issue(
                    "multiple_participant_root_scopes",
                    ("participants", participant_index, "id"),
                    f"Participant '{participant.id}' owns multiple root process scopes at indexes {root_indexes}.",
                )
            )

    for node_index, node in enumerate(model.nodes):
        if node.type != NodeType.SUBPROCESS:
            continue
        child_indexes = child_scopes_by_subprocess.get(node.id, [])
        if not child_indexes:
            issues.append(
                _issue(
                    "missing_subprocess_scope",
                    ("nodes", node_index, "id"),
                    f"Subprocess node '{node.id}' must own exactly one nested process scope.",
                )
            )
        elif len(child_indexes) > 1:
            for duplicate_index in child_indexes[1:]:
                issues.append(
                    _issue(
                        "duplicate_subprocess_scope",
                        ("scopes", duplicate_index, "subprocess_node_id"),
                        f"Subprocess node '{node.id}' already owns nested scope at index {child_indexes[0]}.",
                    )
                )

    completed: set[str] = set()
    for scope in model.scopes:
        if scope.id in completed:
            continue
        path: list[str] = []
        path_positions: dict[str, int] = {}
        current_id: str | None = scope.id
        while current_id is not None and current_id not in completed:
            cycle_start = path_positions.get(current_id)
            if cycle_start is not None:
                for cycle_scope_id in path[cycle_start:]:
                    cycle_scope_index = scope_index_by_id[cycle_scope_id]
                    issues.append(
                        _issue(
                            "scope_parent_cycle",
                            ("scopes", cycle_scope_index, "parent_scope_id"),
                            f"Scope '{cycle_scope_id}' participates in a parent-scope cycle.",
                        )
                    )
                break
            path_positions[current_id] = len(path)
            path.append(current_id)
            current_scope = model.scopes[scope_index_by_id[current_id]]
            parent_id = current_scope.parent_scope_id
            current_id = parent_id if parent_id in scope_index_by_id else None
        completed.update(path)
    return issues


def _validate_references(model: ProcessModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    participant_ids = {item.id for item in model.participants}
    lanes = {item.id: item for item in model.lanes}
    scopes = {item.id: item for item in model.scopes}
    nodes = {item.id: item for item in model.nodes}

    for index, lane in enumerate(model.lanes):
        if lane.participant_id not in participant_ids:
            issues.append(_issue("unknown_participant", ("lanes", index, "participant_id"), f"Participant '{lane.participant_id}' does not exist."))

    for index, scope in enumerate(model.scopes):
        base = ("scopes", index)
        if scope.participant_id not in participant_ids:
            issues.append(_issue("unknown_participant", base + ("participant_id",), f"Participant '{scope.participant_id}' does not exist."))
        nested_fields = (scope.parent_scope_id, scope.subprocess_node_id)
        if (nested_fields[0] is None) != (nested_fields[1] is None):
            issues.append(_issue("invalid_nested_scope", base, "Nested scopes require both parent_scope_id and subprocess_node_id; root scopes require neither."))
        if scope.parent_scope_id is not None and scope.parent_scope_id not in scopes:
            issues.append(_issue("unknown_parent_scope", base + ("parent_scope_id",), f"Scope '{scope.parent_scope_id}' does not exist."))
        if scope.subprocess_node_id is not None:
            owner = nodes.get(scope.subprocess_node_id)
            if owner is None:
                issues.append(_issue("unknown_subprocess_node", base + ("subprocess_node_id",), f"Node '{scope.subprocess_node_id}' does not exist."))
            elif owner.type != NodeType.SUBPROCESS:
                issues.append(_issue("invalid_subprocess_node", base + ("subprocess_node_id",), f"Node '{owner.id}' is not a subprocess."))
            elif owner.scope_id != scope.parent_scope_id:
                issues.append(_issue("invalid_subprocess_parent", base + ("parent_scope_id",), "The subprocess node must belong to the declared parent scope."))
            elif owner.participant_id != scope.participant_id:
                issues.append(_issue("subprocess_participant_mismatch", base + ("participant_id",), "A nested scope and its subprocess node must belong to the same participant."))

    for index, node in enumerate(model.nodes):
        base = ("nodes", index)
        scope = scopes.get(node.scope_id)
        if node.participant_id not in participant_ids:
            issues.append(_issue("unknown_participant", base + ("participant_id",), f"Participant '{node.participant_id}' does not exist."))
        if scope is None:
            issues.append(_issue("unknown_scope", base + ("scope_id",), f"Scope '{node.scope_id}' does not exist."))
        elif node.participant_id != scope.participant_id:
            issues.append(_issue("scope_participant_mismatch", base + ("participant_id",), "Node participant must match its process scope participant."))
        if node.lane_id is not None:
            lane = lanes.get(node.lane_id)
            if lane is None:
                issues.append(_issue("unknown_lane", base + ("lane_id",), f"Lane '{node.lane_id}' does not exist."))
            elif lane.participant_id != node.participant_id:
                issues.append(_issue("lane_participant_mismatch", base + ("lane_id",), "Node and lane must belong to the same participant."))
        if node.default_flow_id is not None and node.type != NodeType.EXCLUSIVE_GATEWAY:
            issues.append(_issue("default_on_non_exclusive_gateway", base + ("default_flow_id",), "Only exclusive gateways may declare a default outgoing flow."))

    for index, flow in enumerate(model.sequence_flows):
        base = ("sequence_flows", index)
        source = nodes.get(flow.source_ref)
        target = nodes.get(flow.target_ref)
        if source is None:
            issues.append(_issue("unknown_sequence_source", base + ("source_ref",), f"Node '{flow.source_ref}' does not exist."))
        if target is None:
            issues.append(_issue("unknown_sequence_target", base + ("target_ref",), f"Node '{flow.target_ref}' does not exist."))
        if source is not None and target is not None:
            if source.participant_id != target.participant_id:
                issues.append(_issue("cross_participant_sequence_flow", base, "Sequence flow cannot connect different participants."))
            if source.scope_id != target.scope_id:
                issues.append(_issue("cross_scope_sequence_flow", base, "Sequence flow cannot cross subprocess scope boundaries."))

    eligible_message_node_types = {NodeType.TASK, NodeType.SUBPROCESS}
    for index, flow in enumerate(model.message_flows):
        base = ("message_flows", index)
        source_node = nodes.get(flow.source_ref)
        target_node = nodes.get(flow.target_ref)
        source_participant = (
            flow.source_ref
            if flow.source_ref in participant_ids
            else source_node.participant_id if source_node is not None else None
        )
        target_participant = (
            flow.target_ref
            if flow.target_ref in participant_ids
            else target_node.participant_id if target_node is not None else None
        )
        if source_participant is None:
            issues.append(_issue("unknown_message_source", base + ("source_ref",), f"Node or participant '{flow.source_ref}' does not exist."))
        elif source_node is not None and source_node.type not in eligible_message_node_types:
            issues.append(_issue("ineligible_message_source", base + ("source_ref",), "Message flow node sources must be tasks or subprocesses in schema 1.0."))
        if target_participant is None:
            issues.append(_issue("unknown_message_target", base + ("target_ref",), f"Node or participant '{flow.target_ref}' does not exist."))
        elif target_node is not None and target_node.type not in eligible_message_node_types:
            issues.append(_issue("ineligible_message_target", base + ("target_ref",), "Message flow node targets must be tasks or subprocesses in schema 1.0."))
        if source_participant is not None and source_participant == target_participant:
            issues.append(_issue("same_participant_message_flow", base, "Message flow endpoints must belong to different participants."))
    return issues


def _validate_scope_graph(model: ProcessModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_by_id = {node.id: node for node in model.nodes}
    flows_by_scope: dict[str, list[tuple[int, SequenceFlow]]] = defaultdict(list)
    for index, flow in enumerate(model.sequence_flows):
        source = node_by_id.get(flow.source_ref)
        target = node_by_id.get(flow.target_ref)
        if source is not None and target is not None and source.scope_id == target.scope_id:
            flows_by_scope[source.scope_id].append((index, flow))

    for scope_index, scope in enumerate(model.scopes):
        scoped_nodes = [node for node in model.nodes if node.scope_id == scope.id]
        scoped_ids = {node.id for node in scoped_nodes}
        starts = {node.id for node in scoped_nodes if node.type == NodeType.START_EVENT}
        ends = {node.id for node in scoped_nodes if node.type == NodeType.END_EVENT}
        if not starts:
            issues.append(_issue("missing_start_event", ("scopes", scope_index), f"Scope '{scope.id}' must contain a start event."))
        if not ends:
            issues.append(_issue("missing_end_event", ("scopes", scope_index), f"Scope '{scope.id}' must contain an end event."))

        outgoing: dict[str, set[str]] = defaultdict(set)
        incoming: dict[str, set[str]] = defaultdict(set)
        flow_records = flows_by_scope.get(scope.id, [])
        for _, flow in flow_records:
            outgoing[flow.source_ref].add(flow.target_ref)
            incoming[flow.target_ref].add(flow.source_ref)
        for node_index, node in enumerate(model.nodes):
            if node.scope_id != scope.id:
                continue
            if node.type == NodeType.START_EVENT and incoming.get(node.id):
                issues.append(_issue("start_has_incoming", ("nodes", node_index), "Start events cannot have incoming sequence flows."))
            if node.type == NodeType.END_EVENT and outgoing.get(node.id):
                issues.append(_issue("end_has_outgoing", ("nodes", node_index), "End events cannot have outgoing sequence flows."))

        reachable_from_start = _reachable(starts, outgoing)
        can_reach_end = _reachable(ends, incoming)
        for node_index, node in enumerate(model.nodes):
            if node.id not in scoped_ids:
                continue
            if starts and node.id not in reachable_from_start:
                issues.append(_issue("unreachable_from_start", ("nodes", node_index), f"Node '{node.id}' is not reachable from a start event in scope '{scope.id}'."))
            if ends and node.id not in can_reach_end:
                issues.append(_issue("cannot_reach_end", ("nodes", node_index), f"Node '{node.id}' cannot reach an end event in scope '{scope.id}'."))
    return issues


def _validate_gateways(model: ProcessModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    outgoing: dict[str, list[tuple[int, SequenceFlow]]] = defaultdict(list)
    for index, flow in enumerate(model.sequence_flows):
        outgoing[flow.source_ref].append((index, flow))

    for node_index, node in enumerate(model.nodes):
        node_outgoing = outgoing.get(node.id, [])
        if node.type == NodeType.EXCLUSIVE_GATEWAY:
            outgoing_ids = {flow.id for _, flow in node_outgoing}
            if node.default_flow_id is not None and node.default_flow_id not in outgoing_ids:
                issues.append(_issue("invalid_gateway_default_flow", ("nodes", node_index, "default_flow_id"), "Default flow must reference an outgoing sequence flow of this exclusive gateway."))
            for flow_index, flow in node_outgoing:
                if flow.id == node.default_flow_id:
                    if flow.condition is not None:
                        issues.append(_issue("condition_on_default_flow", ("sequence_flows", flow_index, "condition"), "An exclusive gateway default flow cannot have a condition."))
                elif len(node_outgoing) > 1 and flow.condition is None:
                    issues.append(_issue("missing_gateway_condition", ("sequence_flows", flow_index, "condition"), "Each non-default branch from an exclusive gateway requires a condition label."))
        elif node.type == NodeType.PARALLEL_GATEWAY:
            for flow_index, flow in node_outgoing:
                if flow.condition is not None:
                    issues.append(_issue("parallel_gateway_condition", ("sequence_flows", flow_index, "condition"), "Parallel gateway outgoing flows cannot declare conditions in schema 1.0."))
    return issues


def validate_process_graph(model: ProcessModel) -> None:
    """Validate references and graph invariants on an already parsed model.

    All detectable issues are returned together in stable collection order.
    The input is never changed or repaired.
    """
    issues = [
        *_validate_unique_ids(model),
        *_validate_scope_hierarchy(model),
        *_validate_references(model),
        *_validate_scope_graph(model),
        *_validate_gateways(model),
    ]
    if issues:
        raise ProcessGraphValidationError(issues)
