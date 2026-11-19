"""Deterministically generate coupled English/ProcessModel training references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ProcessModel, validate_process_graph
from app.domain.process_model import NodeType
from training.build_supervised_dataset import RESERVED_CORPUS, _assert_not_reserved, _load_reserved, _normalize

DEFAULT_REFERENCES = ROOT / "training" / "artifacts" / "synthetic_references.json"


@dataclass(frozen=True, slots=True)
class Domain:
    key: str
    participant: str
    object_name: str
    actions: tuple[str, ...]
    conditions: tuple[str, ...]


DOMAINS = (
    Domain("invoice", "Accounts Team", "invoice", ("Register invoice", "Check totals", "Verify tax details", "Match purchase record", "Authorize payment", "Archive invoice"), ("Invoice is compliant", "Amount exceeds approval limit", "Supporting record is complete")),
    Domain("shipment", "Logistics Team", "shipment", ("Register shipment", "Check destination", "Reserve capacity", "Prepare documents", "Schedule dispatch", "Confirm delivery"), ("Destination is serviceable", "Capacity is available", "Customs data is complete")),
    Domain("maintenance", "Maintenance Team", "work order", ("Log work order", "Assess equipment", "Reserve technician", "Prepare tools", "Perform repair", "Record completion"), ("Equipment is safe", "Specialist support is required", "Parts are available")),
    Domain("application", "Admissions Office", "application", ("Register application", "Check identity", "Review evidence", "Verify eligibility", "Record decision", "Notify applicant"), ("Application is eligible", "Evidence is complete", "Additional review is required")),
    Domain("contract", "Contract Office", "contract", ("Register contract", "Review clauses", "Check authority", "Assess obligations", "Record approval", "Archive contract"), ("Contract is acceptable", "Legal escalation is required", "Signatures are complete")),
    Domain("inventory", "Inventory Team", "stock request", ("Register stock request", "Check inventory", "Reserve items", "Prepare stock", "Update ledger", "Close request"), ("Stock is available", "Replenishment is required", "Quantity is authorized")),
    Domain("permit", "Permit Office", "permit request", ("Register permit request", "Check documents", "Inspect location", "Assess compliance", "Issue decision", "Archive request"), ("Request is compliant", "Inspection is required", "Documents are complete")),
    Domain("support", "Support Desk", "support case", ("Register support case", "Classify issue", "Collect diagnostics", "Test resolution", "Record outcome", "Close case"), ("Resolution is successful", "Escalation is required", "Diagnostics are complete")),
    Domain("quality", "Quality Team", "quality case", ("Register quality case", "Inspect sample", "Analyze measurements", "Check tolerance", "Record finding", "Release batch"), ("Sample meets tolerance", "Rework is required", "Measurements are complete")),
    Domain("training", "Learning Office", "training request", ("Register training request", "Check prerequisites", "Assign instructor", "Prepare materials", "Record attendance", "Close training"), ("Prerequisites are met", "Manager approval is required", "Materials are ready")),
    Domain("security", "Security Team", "access case", ("Register access case", "Verify identity", "Assess privileges", "Check conflicts", "Record authorization", "Close access case"), ("Access is justified", "Conflict is detected", "Identity is verified")),
    Domain("procurement", "Procurement Team", "purchase request", ("Register purchase request", "Check specification", "Compare offers", "Verify budget", "Record selection", "Close purchase request"), ("Budget is available", "Competitive review is required", "Specification is complete")),
)


def topology_fingerprint(model: ProcessModel) -> str:
    """Return a vocabulary/ID-independent graph fingerprint with scope ownership."""
    nodes = {node.id: node for node in model.nodes}
    scopes = {scope.id: scope for scope in model.scopes}
    scope_depth: dict[str, int] = {}
    for scope in model.scopes:
        depth, current = 0, scope
        while current.parent_scope_id is not None:
            depth += 1
            current = scopes[current.parent_scope_id]
        scope_depth[scope.id] = depth
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for flow in model.sequence_flows:
        outgoing[flow.source_ref].append(flow.target_ref)
        incoming[flow.target_ref].append(flow.source_ref)
    labels = {node.id: f"{node.type.value}|d{scope_depth[node.scope_id]}" for node in model.nodes}
    for _ in range(len(model.nodes)):
        updated = {
            node_id: hashlib.sha256(
                (labels[node_id] + "<" + ",".join(sorted(labels[item] for item in incoming[node_id])) + ">" + ",".join(sorted(labels[item] for item in outgoing[node_id]))).encode()
            ).hexdigest()[:20]
            for node_id in labels
        }
        labels = updated
    payload = {
        "participants": len(model.participants),
        "lanes_per_participant": sorted(sum(lane.participant_id == participant.id for lane in model.lanes) for participant in model.participants),
        "scopes": sorted((scope_depth[scope.id], scope.subprocess_node_id is not None) for scope in model.scopes),
        "nodes": sorted(labels.values()),
        "sequence_edges": sorted((labels[flow.source_ref], labels[flow.target_ref], flow.condition is not None, nodes[flow.source_ref].default_flow_id == flow.id) for flow in model.sequence_flows),
        "message_edges": sorted((labels.get(flow.source_ref, "participant"), labels.get(flow.target_ref, "participant")) for flow in model.message_flows),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Builder:
    def __init__(self, case_id: str, name: str) -> None:
        self.case_id, self.name = case_id, name
        self.participants: list[dict[str, str]] = []
        self.scopes: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.flows: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self._flow = 0

    def participant(self, key: str, name: str) -> tuple[str, str]:
        participant, scope = f"p_{key}", f"s_{key}"
        self.participants.append({"id": participant, "name": name})
        self.scopes.append({"id": scope, "participant_id": participant, "parent_scope_id": None, "subprocess_node_id": None})
        return participant, scope

    def node(self, key: str, kind: NodeType, name: str, participant: str, scope: str, *, default: str | None = None) -> str:
        node_id = f"n_{key}"
        self.nodes.append({"id": node_id, "type": kind.value, "name": name, "participant_id": participant, "lane_id": None, "scope_id": scope, "default_flow_id": default})
        return node_id

    def flow(self, source: str, target: str, condition: str | None = None, *, flow_id: str | None = None) -> str:
        self._flow += 1
        identifier = flow_id or f"f_{self._flow}"
        self.flows.append({"id": identifier, "source_ref": source, "target_ref": target, "condition": condition})
        return identifier

    def chain(self, values: Iterable[str]) -> None:
        values = list(values)
        for source, target in zip(values, values[1:]):
            self.flow(source, target)

    def model(self) -> ProcessModel:
        model = ProcessModel.model_validate({"schema_version": "1.0", "id": f"model_{self.case_id}", "name": self.name, "participants": self.participants, "lanes": [], "scopes": self.scopes, "nodes": self.nodes, "sequence_flows": self.flows, "message_flows": self.messages})
        validate_process_graph(model)
        return model


def _actions(domain: Domain, count: int, offset: int = 0) -> list[str]:
    return [domain.actions[(offset + index) % len(domain.actions)] + (f" stage {index + 1}" if count > len(domain.actions) else "") for index in range(count)]


def sequence_case(domain: Domain, count: int) -> tuple[str, ProcessModel, str]:
    key = f"sequence_t{count}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} sequence")
    participant, scope = builder.participant("owner", domain.participant)
    names = _actions(domain, count)
    values = [builder.node("start", NodeType.START_EVENT, f"{domain.object_name.title()} received", participant, scope)]
    values += [builder.node(f"task_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(names)]
    values += [builder.node("end", NodeType.END_EVENT, f"{domain.object_name.title()} completed", participant, scope)]
    builder.chain(values)
    return f"{domain.participant} handles the {domain.object_name}. It " + ", then ".join(name.lower() for name in names) + f", and completes the {domain.object_name}.", builder.model(), key


def xor_case(domain: Domain, prefix: int, branches: int, branch_length: int) -> tuple[str, ProcessModel, str]:
    key = f"xor_p{prefix}_b{branches}_l{branch_length}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} decision")
    participant, scope = builder.participant("owner", domain.participant)
    start = builder.node("start", NodeType.START_EVENT, f"{domain.object_name.title()} received", participant, scope)
    prefix_names = _actions(domain, prefix)
    prefix_ids = [builder.node(f"pre_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(prefix_names)]
    split = builder.node("split", NodeType.EXCLUSIVE_GATEWAY, f"Route {domain.object_name}", participant, scope, default="f_default")
    builder.chain([start, *prefix_ids, split])
    merge = builder.node("merge", NodeType.EXCLUSIVE_GATEWAY, f"Merge {domain.object_name} outcomes", participant, scope)
    phrases = []
    for branch in range(branches):
        names = [f"{name} option {branch + 1}" for name in _actions(domain, branch_length, branch + prefix)]
        ids = [builder.node(f"b{branch}_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(names)]
        condition = domain.conditions[branch % len(domain.conditions)]
        builder.flow(split, ids[0], None if branch == branches - 1 else condition, flow_id="f_default" if branch == branches - 1 else None)
        builder.chain(ids)
        builder.flow(ids[-1], merge)
        phrases.append(("otherwise" if branch == branches - 1 else f"if {condition.lower()}") + ", it " + " then ".join(name.lower() for name in names))
    end = builder.node("end", NodeType.END_EVENT, f"{domain.object_name.title()} closed", participant, scope)
    builder.flow(merge, end)
    description = f"{domain.participant} " + " then ".join(name.lower() for name in prefix_names) + ". " + "; ".join(phrases) + f". The alternatives merge before the {domain.object_name} is closed."
    return description, builder.model(), key


def parallel_case(domain: Domain, prefix: int, branches: int, branch_length: int) -> tuple[str, ProcessModel, str]:
    key = f"parallel_p{prefix}_b{branches}_l{branch_length}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} parallel work")
    participant, scope = builder.participant("owner", domain.participant)
    start = builder.node("start", NodeType.START_EVENT, f"{domain.object_name.title()} received", participant, scope)
    prefix_names = _actions(domain, prefix)
    prefix_ids = [builder.node(f"pre_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(prefix_names)]
    split = builder.node("split", NodeType.PARALLEL_GATEWAY, "Start parallel work", participant, scope)
    builder.chain([start, *prefix_ids, split])
    join = builder.node("join", NodeType.PARALLEL_GATEWAY, "Join parallel work", participant, scope)
    phrases = []
    for branch in range(branches):
        names = [f"{name} branch {branch + 1}" for name in _actions(domain, branch_length, branch + prefix)]
        ids = [builder.node(f"b{branch}_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(names)]
        builder.flow(split, ids[0]); builder.chain(ids); builder.flow(ids[-1], join)
        phrases.append(" then ".join(name.lower() for name in names))
    end = builder.node("end", NodeType.END_EVENT, f"{domain.object_name.title()} completed", participant, scope)
    builder.flow(join, end)
    description = f"{domain.participant} " + " then ".join(name.lower() for name in prefix_names) + ". It starts these branches in parallel: " + "; ".join(phrases) + f". It waits for every branch before completing the {domain.object_name}."
    return description, builder.model(), key


def loop_case(domain: Domain, prefix: int, body: int, exit_tasks: int) -> tuple[str, ProcessModel, str]:
    key = f"loop_p{prefix}_b{body}_e{exit_tasks}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} revision loop")
    participant, scope = builder.participant("owner", domain.participant)
    start = builder.node("start", NodeType.START_EVENT, f"{domain.object_name.title()} received", participant, scope)
    prefix_names = _actions(domain, prefix)
    prefix_ids = [builder.node(f"pre_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(prefix_names)]
    review = builder.node("review", NodeType.TASK, f"Review {domain.object_name}", participant, scope)
    decision = builder.node("decision", NodeType.EXCLUSIVE_GATEWAY, f"{domain.object_name.title()} acceptable", participant, scope, default="f_accept")
    builder.chain([start, *prefix_ids, review, decision])
    body_names = [f"{name} correction" for name in _actions(domain, body, prefix + 1)]
    body_ids = [builder.node(f"redo_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(body_names)]
    builder.flow(decision, body_ids[0], domain.conditions[1]); builder.chain(body_ids); builder.flow(body_ids[-1], review)
    exit_names = [f"{name} final" for name in _actions(domain, exit_tasks, prefix + body + 1)]
    exit_ids = [builder.node(f"exit_{i}", NodeType.TASK, name, participant, scope) for i, name in enumerate(exit_names)]
    builder.flow(decision, exit_ids[0], flow_id="f_accept"); builder.chain(exit_ids)
    end = builder.node("end", NodeType.END_EVENT, f"{domain.object_name.title()} completed", participant, scope); builder.flow(exit_ids[-1], end)
    description = f"{domain.participant} " + " then ".join(name.lower() for name in prefix_names) + f" and performs Review {domain.object_name}. If {domain.conditions[1].lower()}, it " + " then ".join(name.lower() for name in body_names) + f" and returns to Review {domain.object_name}. Otherwise it " + " then ".join(name.lower() for name in exit_names) + f" and completes the {domain.object_name}."
    return description, builder.model(), key


def collaboration_case(domain: Domain, sender_tasks: int, receiver_tasks: int, messages: int) -> tuple[str, ProcessModel, str]:
    key = f"collaboration_s{sender_tasks}_r{receiver_tasks}_m{messages}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} collaboration")
    sender, sender_scope = builder.participant("sender", domain.participant)
    receiver_name = f"{domain.object_name.title()} Service"
    receiver, receiver_scope = builder.participant("receiver", receiver_name)
    sender_names = [f"{name} sender" for name in _actions(domain, sender_tasks)]
    receiver_names = [f"{name} receiver" for name in _actions(domain, receiver_tasks, 2)]
    sender_ids = [builder.node("s_start", NodeType.START_EVENT, "Sender ready", sender, sender_scope)] + [builder.node(f"s_{i}", NodeType.TASK, name, sender, sender_scope) for i, name in enumerate(sender_names)] + [builder.node("s_end", NodeType.END_EVENT, "Sender completed", sender, sender_scope)]
    receiver_ids = [builder.node("r_start", NodeType.START_EVENT, "Receiver ready", receiver, receiver_scope)] + [builder.node(f"r_{i}", NodeType.TASK, name, receiver, receiver_scope) for i, name in enumerate(receiver_names)] + [builder.node("r_end", NodeType.END_EVENT, "Receiver completed", receiver, receiver_scope)]
    builder.chain(sender_ids); builder.chain(receiver_ids)
    phrases = []
    for index in range(messages):
        label = f"{domain.object_name.title()} message {index + 1}"
        source_name, target_name = sender_names[index % sender_tasks], receiver_names[index % receiver_tasks]
        builder.messages.append({"id": f"m_{index}", "source_ref": sender_ids[1 + index % sender_tasks], "target_ref": receiver_ids[1 + index % receiver_tasks], "label": label})
        phrases.append(f"{label} goes from {source_name} to {target_name}")
    description = f"{domain.participant} performs " + " then ".join(sender_names) + f". {receiver_name} performs " + " then ".join(receiver_names) + ". " + ". ".join(phrases) + ". Each participant keeps its own sequence flow."
    return description, builder.model(), key


def subprocess_case(domain: Domain, parent_prefix: int, inner_tasks: int, inner_kind: str) -> tuple[str, ProcessModel, str]:
    key = f"subprocess_p{parent_prefix}_i{inner_tasks}_{inner_kind}"
    builder = Builder(f"{domain.key}_{key}", f"{domain.object_name.title()} subprocess")
    participant, root = builder.participant("owner", domain.participant)
    start = builder.node("start", NodeType.START_EVENT, f"{domain.object_name.title()} received", participant, root)
    prefix_names = _actions(domain, parent_prefix)
    prefix_ids = [builder.node(f"pre_{i}", NodeType.TASK, name, participant, root) for i, name in enumerate(prefix_names)]
    subprocess_name = f"Perform detailed {domain.object_name} work"
    subprocess = builder.node("subprocess", NodeType.SUBPROCESS, subprocess_name, participant, root)
    end = builder.node("end", NodeType.END_EVENT, f"{domain.object_name.title()} completed", participant, root)
    builder.chain([start, *prefix_ids, subprocess, end])
    nested = "s_nested"
    builder.scopes.append({"id": nested, "participant_id": participant, "parent_scope_id": root, "subprocess_node_id": subprocess})
    inner_start = builder.node("inner_start", NodeType.START_EVENT, "Detailed work started", participant, nested)
    names = [f"{name} internally" for name in _actions(domain, inner_tasks, parent_prefix)]
    if inner_kind == "sequence":
        ids = [builder.node(f"inner_{i}", NodeType.TASK, name, participant, nested) for i, name in enumerate(names)]
        inner_end = builder.node("inner_end", NodeType.END_EVENT, "Detailed work completed", participant, nested)
        builder.chain([inner_start, *ids, inner_end])
        inner_text = " then ".join(names)
    else:
        kind = NodeType.PARALLEL_GATEWAY if inner_kind == "parallel" else NodeType.EXCLUSIVE_GATEWAY
        split = builder.node("inner_split", kind, "Route internal work", participant, nested, default="f_inner_default" if inner_kind == "xor" else None)
        builder.flow(inner_start, split)
        join = builder.node("inner_join", kind, "Join internal work", participant, nested)
        for index, name in enumerate(names):
            task = builder.node(f"inner_{index}", NodeType.TASK, name, participant, nested)
            builder.flow(split, task, None if inner_kind == "parallel" or index == len(names) - 1 else domain.conditions[index % len(domain.conditions)], flow_id="f_inner_default" if inner_kind == "xor" and index == len(names) - 1 else None)
            builder.flow(task, join)
        inner_end = builder.node("inner_end", NodeType.END_EVENT, "Detailed work completed", participant, nested); builder.flow(join, inner_end)
        inner_text = ("in parallel: " if inner_kind == "parallel" else "as conditional alternatives: ") + ", ".join(names)
        if inner_kind == "xor":
            inner_text += ". The alternatives use these conditions: " + ", ".join(
                domain.conditions[index % len(domain.conditions)]
                for index in range(len(names) - 1)
            )
    description = f"{domain.participant} " + " then ".join(prefix_names) + f" before executing the subprocess {subprocess_name}. Inside {subprocess_name}, detailed work starts and the team performs {inner_text}; then the internal scope ends. After the subprocess returns, the {domain.object_name} is completed."
    return description, builder.model(), key


def _candidate_stream() -> Iterable[tuple[str, ProcessModel, str, str]]:
    index = 0
    generators: list[tuple[str, Any, Iterable[tuple[Any, ...]]]] = [
        ("sequence", sequence_case, ((count,) for count in range(2, 15))),
        ("exclusive_decision", xor_case, ((p, b, length) for p in range(1, 6) for b in range(2, 6) for length in range(1, 5))),
        ("parallel", parallel_case, ((p, b, length) for p in range(1, 6) for b in range(2, 6) for length in range(1, 5))),
        ("loop", loop_case, ((p, body, exit_tasks) for p in range(1, 6) for body in range(1, 6) for exit_tasks in range(1, 5))),
        ("collaboration", collaboration_case, ((sender, receiver, messages) for sender in range(1, 7) for receiver in range(1, 7) for messages in range(1, min(4, sender * receiver) + 1))),
        ("subprocess", subprocess_case, ((parent, inner, kind) for parent in range(1, 6) for inner in range(2, 7) for kind in ("sequence", "parallel", "xor"))),
    ]
    for family, generator, parameters in generators:
        for args in parameters:
            domain = DOMAINS[index % len(DOMAINS)]; index += 1
            description, model, structure_family = generator(domain, *args)
            yield description, model, family, structure_family


def _split_group(family: str, structure_family: str) -> str:
    if family == "sequence":
        count = int(structure_family.rsplit("t", 1)[1])
        return f"sequence_{'short' if count <= 5 else 'medium' if count <= 9 else 'long'}"
    match_key = {
        "exclusive_decision": r"_b(\d+)",
        "parallel": r"_b(\d+)",
        "loop": r"_b(\d+)",
        "collaboration": r"_m(\d+)",
    }.get(family)
    if match_key:
        import re
        return f"{family}_{re.search(match_key, structure_family).group(1)}"
    return f"subprocess_{structure_family.rsplit('_', 1)[1]}"


def generate_references() -> tuple[dict[str, Any], dict[str, Any]]:
    reserved_descriptions, reserved_semantic, _ = _load_reserved()
    corpus = json.loads(RESERVED_CORPUS.read_text(encoding="utf-8"))
    reserved_topologies = {
        topology_fingerprint(ProcessModel.model_validate(json.loads((ROOT / case["reference"]).read_text(encoding="utf-8"))))
        for case in corpus["cases"]
    }
    cases: list[dict[str, Any]] = []
    seen_topologies: set[str] = set()
    seen_descriptions: set[str] = set()
    rejections: Counter[str] = Counter()
    for description, model, family, structure_family in _candidate_stream():
        topology = topology_fingerprint(model)
        try:
            _assert_not_reserved({"id": model.id, "description": description}, model, reserved_descriptions, reserved_semantic)
            if topology in reserved_topologies:
                raise ValueError("reserved_topology_equivalent")
            if topology in seen_topologies:
                raise ValueError("duplicate_topology")
            normalized = _normalize(description)
            if normalized in seen_descriptions:
                raise ValueError("duplicate_description")
            required = [participant.name for participant in model.participants]
            required += [node.name for node in model.nodes if node.type in {NodeType.TASK, NodeType.SUBPROCESS}]
            required += [flow.condition for flow in model.sequence_flows if flow.condition]
            required += [flow.label for flow in model.message_flows if flow.label]
            if any(_normalize(item) not in _normalize(description) for item in required):
                raise ValueError("text_graph_element_missing")
        except ValueError as error:
            rejections[str(error).split(":", 1)[-1].strip()] += 1
            continue
        seen_topologies.add(topology); seen_descriptions.add(normalized)
        cases.append({
            "id": model.id.removeprefix("model_"), "family": family, "structure_family": structure_family,
            "split_group": _split_group(family, structure_family),
            "description": description,
            "provenance": {"kind": "synthetic_grammar", "synthetic": True, "source": "deterministic_business_process_grammar_v1", "generator_version": "1.0"},
            "topology_fingerprint": topology, "model": model.model_dump(mode="json"),
        })
    report = {
        "candidate_examples": sum(rejections.values()) + len(cases), "accepted_examples": len(cases),
        "rejected_examples": sum(rejections.values()), "rejection_reasons": dict(sorted(rejections.items())),
        "families": dict(sorted(Counter(case["family"] for case in cases).items())),
        "complexity": dict(sorted(Counter("simple" if len(case["model"]["nodes"]) <= 8 else "medium" if len(case["model"]["nodes"]) <= 16 else "complex" for case in cases).items())),
        "unique_topologies": len(seen_topologies), "reserved_evaluation_cases": 12,
    }
    return {"dataset_version": "synthetic-grammar-1.0", "cases": cases}, report


def write_references(output: Path = DEFAULT_REFERENCES) -> dict[str, Any]:
    document, report = generate_references()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    (output.parent / "synthetic_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic ProcessModel references")
    parser.add_argument("--output", type=Path, default=DEFAULT_REFERENCES)
    args = parser.parse_args()
    print(json.dumps(write_references(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
