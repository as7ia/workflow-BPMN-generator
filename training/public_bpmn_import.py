"""Research-only importer for licensed public text/BPMN pairs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ProcessModel, validate_process_graph
from app.domain.process_model import NodeType
from training.build_supervised_dataset import (
    _load_reserved,
    _normalize,
    _semantic_fingerprint,
)

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
SUPPORTED_TASKS = {
    "task", "userTask", "serviceTask", "manualTask", "scriptTask",
    "businessRuleTask", "sendTask", "receiveTask",
}
SUPPORTED_NODES = SUPPORTED_TASKS | {
    "startEvent", "endEvent", "exclusiveGateway", "parallelGateway", "subProcess"
}
UNSUPPORTED_FLOW_NODES = {
    "inclusiveGateway", "eventBasedGateway", "complexGateway", "intermediateCatchEvent",
    "intermediateThrowEvent", "boundaryEvent", "callActivity", "transaction", "adHocSubProcess",
    "messageEventDefinition", "timerEventDefinition", "conditionalEventDefinition",
    "signalEventDefinition", "errorEventDefinition", "escalationEventDefinition",
    "compensateEventDefinition", "cancelEventDefinition", "terminateEventDefinition",
    "linkEventDefinition", "textAnnotation", "association", "dataObjectReference",
    "dataStoreReference", "group",
}
CAMUNDA_LICENSE = {
    "name": "Camunda BPMN for Research terms",
    "research_only": True,
    "commercial_use": False,
    "attribution_required": True,
    "source_url": "https://github.com/camunda/bpmn-for-research",
}


class ImportRejection(ValueError):
    """A source pair cannot be represented faithfully in ProcessModel 1.0."""


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _safe_id(prefix: str, raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    return f"{prefix}_{cleaned}"


def _required_name(element: ET.Element, kind: str) -> str:
    value = " ".join(element.get("name", "").split())
    if not value:
        raise ImportRejection(f"missing_business_name:{kind}")
    return value


def bpmn_to_process_model(xml: str, *, source_id: str) -> ProcessModel:
    """Convert the lossless ProcessModel 1.0 subset or reject the whole model."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ImportRejection("malformed_bpmn_xml") from error
    for element in root.iter():
        if _local(element) in UNSUPPORTED_FLOW_NODES:
            raise ImportRejection(f"unsupported_bpmn_element:{_local(element)}")

    processes = {
        item.get("id"): item
        for item in root.findall(f"{{{BPMN_NS}}}process")
        if item.get("id")
    }
    collaboration = root.find(f"{{{BPMN_NS}}}collaboration")
    participant_specs: list[tuple[str, str, str]] = []
    if collaboration is not None:
        for item in collaboration.findall(f"{{{BPMN_NS}}}participant"):
            process_ref = item.get("processRef")
            if not process_ref or process_ref not in processes:
                raise ImportRejection("participant_without_supported_process")
            participant_specs.append((item.get("id", ""), _required_name(item, "participant"), process_ref))
    elif len(processes) == 1:
        process_ref, process = next(iter(processes.items()))
        participant_specs.append((f"participant_{process_ref}", _required_name(process, "process"), process_ref))
    else:
        raise ImportRejection("missing_collaboration_participants")

    participants: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    scopes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    sequence_flows: list[dict[str, Any]] = []
    message_flows: list[dict[str, Any]] = []
    raw_node_to_id: dict[str, str] = {}
    raw_participant_to_id: dict[str, str] = {}
    raw_flow_to_id: dict[str, str] = {}

    for raw_participant, participant_name, process_ref in participant_specs:
        participant_id = _safe_id("participant", raw_participant)
        raw_participant_to_id[raw_participant] = participant_id
        participants.append({"id": participant_id, "name": participant_name})
        process = processes[process_ref]
        root_scope_id = _safe_id("scope", process_ref)
        scopes.append({
            "id": root_scope_id,
            "participant_id": participant_id,
            "parent_scope_id": None,
            "subprocess_node_id": None,
        })
        lane_by_node: dict[str, str] = {}
        for lane in process.findall(f".//{{{BPMN_NS}}}lane"):
            lane_id = _safe_id("lane", lane.get("id", ""))
            lanes.append({
                "id": lane_id,
                "name": _required_name(lane, "lane"),
                "participant_id": participant_id,
            })
            for reference in lane.findall(f"{{{BPMN_NS}}}flowNodeRef"):
                if reference.text:
                    lane_by_node[reference.text.strip()] = lane_id

        def parse_container(container: ET.Element, scope_id: str) -> None:
            elements = list(container)
            for element in elements:
                kind = _local(element)
                if kind not in SUPPORTED_NODES:
                    continue
                raw_id = element.get("id")
                if not raw_id:
                    raise ImportRejection(f"missing_id:{kind}")
                node_id = _safe_id("node", raw_id)
                if raw_id in raw_node_to_id:
                    raise ImportRejection("duplicate_bpmn_node_id")
                raw_node_to_id[raw_id] = node_id
                if kind in SUPPORTED_TASKS or kind == "subProcess":
                    name = _required_name(element, kind)
                elif kind == "startEvent":
                    name = " ".join(element.get("name", "Start").split()) or "Start"
                elif kind == "endEvent":
                    name = " ".join(element.get("name", "End").split()) or "End"
                else:
                    name = " ".join(element.get("name", kind).split()) or kind
                node_type = {
                    "startEvent": NodeType.START_EVENT.value,
                    "endEvent": NodeType.END_EVENT.value,
                    "exclusiveGateway": NodeType.EXCLUSIVE_GATEWAY.value,
                    "parallelGateway": NodeType.PARALLEL_GATEWAY.value,
                    "subProcess": NodeType.SUBPROCESS.value,
                }.get(kind, NodeType.TASK.value)
                default_raw = element.get("default")
                nodes.append({
                    "id": node_id,
                    "type": node_type,
                    "name": name,
                    "participant_id": participant_id,
                    "lane_id": lane_by_node.get(raw_id),
                    "scope_id": scope_id,
                    "default_flow_id": _safe_id("flow", default_raw) if default_raw else None,
                })
                if kind == "subProcess":
                    nested_scope_id = _safe_id("scope", raw_id)
                    scopes.append({
                        "id": nested_scope_id,
                        "participant_id": participant_id,
                        "parent_scope_id": scope_id,
                        "subprocess_node_id": node_id,
                    })
                    parse_container(element, nested_scope_id)
            for flow in container.findall(f"{{{BPMN_NS}}}sequenceFlow"):
                raw_id = flow.get("id")
                source = flow.get("sourceRef")
                target = flow.get("targetRef")
                if not raw_id or not source or not target:
                    raise ImportRejection("incomplete_sequence_flow")
                raw_flow_to_id[raw_id] = _safe_id("flow", raw_id)
                condition_element = flow.find(f"{{{BPMN_NS}}}conditionExpression")
                condition = " ".join((flow.get("name") or (condition_element.text if condition_element is not None else "") or "").split()) or None
                sequence_flows.append({
                    "id": raw_flow_to_id[raw_id],
                    "source_ref": _safe_id("node", source),
                    "target_ref": _safe_id("node", target),
                    "condition": condition,
                })

        parse_container(process, root_scope_id)

    node_by_id = {node["id"]: node for node in nodes}
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for flow in sequence_flows:
        outgoing.setdefault(flow["source_ref"], []).append(flow)
    for node in nodes:
        if node["type"] != NodeType.EXCLUSIVE_GATEWAY.value:
            continue
        branches = outgoing.get(node["id"], [])
        if len(branches) > 1:
            for flow in branches:
                if flow["id"] != node["default_flow_id"] and flow["condition"] is None:
                    raise ImportRejection("missing_explicit_xor_condition")

    if collaboration is not None:
        for flow in collaboration.findall(f"{{{BPMN_NS}}}messageFlow"):
            source_raw = flow.get("sourceRef", "")
            target_raw = flow.get("targetRef", "")
            source = raw_participant_to_id.get(source_raw, raw_node_to_id.get(source_raw))
            target = raw_participant_to_id.get(target_raw, raw_node_to_id.get(target_raw))
            if source is None or target is None:
                raise ImportRejection("unknown_message_endpoint")
            for endpoint in (source, target):
                if endpoint in node_by_id and node_by_id[endpoint]["type"] not in {
                    NodeType.TASK.value, NodeType.SUBPROCESS.value
                }:
                    raise ImportRejection("unsupported_message_event_endpoint")
            message_flows.append({
                "id": _safe_id("message", flow.get("id", "")),
                "source_ref": source,
                "target_ref": target,
                "label": " ".join(flow.get("name", "").split()) or None,
            })

    model = ProcessModel.model_validate({
        "schema_version": "1.0",
        "id": _safe_id("model", source_id),
        "name": source_id.replace("-", " "),
        "participants": participants,
        "lanes": lanes,
        "scopes": scopes,
        "nodes": nodes,
        "sequence_flows": sequence_flows,
        "message_flows": message_flows,
    })
    validate_process_graph(model)
    return model


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("pypdf is required to read Camunda exercise descriptions") from error
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _activity_coverage(description: str, model: ProcessModel) -> float:
    description_words = set(_normalize(description).split())
    activities = [
        set(_normalize(node.name).split())
        for node in model.nodes
        if node.type in {NodeType.TASK, NodeType.SUBPROCESS}
    ]
    if not activities:
        return 0.0
    return sum(bool(words & description_words) for words in activities) / len(activities)


def import_camunda(
    source_root: Path,
    *,
    accept_research_only_license: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not accept_research_only_license:
        raise ValueError("Camunda research-only, attribution, and non-commercial terms must be accepted")
    english = source_root / "BPMN for Research" / "English"
    if not english.is_dir():
        matches = list(source_root.glob("**/BPMN for Research/English"))
        if len(matches) != 1:
            raise ValueError("Cannot locate the Camunda English dataset root")
        english = matches[0]
    reserved_descriptions, reserved_fingerprints, _ = _load_reserved()
    seen_descriptions: set[str] = set()
    seen_models: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    candidates = 0
    for case_dir in sorted(path for path in english.iterdir() if path.is_dir()):
        pdfs = list((case_dir / "01-Exercise").glob("*.pdf"))
        solutions = list((case_dir / "03-Solution").glob("*.bpmn"))
        if len(pdfs) != 1:
            rejected.append({"id": case_dir.name, "reason": "missing_unique_english_description"})
            continue
        description = _extract_pdf_text(pdfs[0])
        for solution in solutions:
            candidates += 1
            pair_id = f"camunda_{case_dir.name}_{solution.stem}"
            try:
                model = bpmn_to_process_model(solution.read_text(encoding="utf-8"), source_id=pair_id)
                normalized = _normalize(description)
                fingerprint = _semantic_fingerprint(model)
                if max(SequenceMatcher(None, normalized, item).ratio() for item in reserved_descriptions) >= 0.72:
                    raise ImportRejection("reserved_evaluation_description_equivalent")
                if fingerprint in reserved_fingerprints:
                    raise ImportRejection("reserved_evaluation_model_equivalent")
                if normalized in seen_descriptions:
                    raise ImportRejection("duplicate_description")
                if fingerprint in seen_models:
                    raise ImportRejection("duplicate_model")
                coverage = _activity_coverage(description, model)
                if coverage < 0.5:
                    raise ImportRejection("text_model_activity_coverage_below_0.5")
            except (ImportRejection, ValueError) as error:
                rejected.append({"id": pair_id, "reason": str(error).split("\n", 1)[0]})
                continue
            seen_descriptions.add(normalized)
            seen_models.add(fingerprint)
            accepted.append({
                "id": pair_id,
                "family": _family(model),
                "description": description,
                "provenance": {
                    "kind": "public_human_authored_research_dataset",
                    "synthetic": False,
                    "source": CAMUNDA_LICENSE["source_url"],
                    "source_file": str(solution.relative_to(source_root)),
                    "license": CAMUNDA_LICENSE,
                    "activity_name_coverage": coverage,
                },
                "model": model.model_dump(mode="json"),
            })
    report = {
        "source": "camunda/bpmn-for-research",
        "source_root": str(source_root.resolve()),
        "accessibility": "downloaded_and_inspected",
        "license": CAMUNDA_LICENSE,
        "candidate_pairs": candidates,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "rejections": rejected,
        "families": dict(sorted(Counter(item["family"] for item in accepted).items())),
        "complexity": dict(sorted(Counter(_complexity(ProcessModel.model_validate(item["model"])) for item in accepted).items())),
    }
    return accepted, report


def _family(model: ProcessModel) -> str:
    features = []
    if len(model.participants) > 1 or model.message_flows:
        features.append("collaboration")
    if any(node.type == NodeType.SUBPROCESS for node in model.nodes):
        features.append("subprocess")
    if any(node.type == NodeType.PARALLEL_GATEWAY for node in model.nodes):
        features.append("parallel")
    if any(node.type == NodeType.EXCLUSIVE_GATEWAY for node in model.nodes):
        features.append("exclusive")
    return "+".join(features) or "linear"


def _complexity(model: ProcessModel) -> str:
    score = len(model.nodes) + 2 * sum(node.type.value.endswith("gateway") for node in model.nodes) + 3 * len(model.message_flows) + 3 * (len(model.scopes) - len(model.participants))
    return "simple" if score <= 8 else "medium" if score <= 20 else "complex"


def write_import(output: Path, cases: list[dict[str, Any]], report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "camunda_reference_cases.json").write_text(
        json.dumps({"dataset_version": "camunda-research-import-1", "cases": cases}, indent=2),
        encoding="utf-8",
    )
    source_report = {
        "sources": [
            report,
            {
                "source": "zenodo:7783492",
                "accessibility": "downloaded_md5_verified",
                "artifact": "text_and_bpmn.raw.tgz",
                "verified_md5": "a0ce779baefc9667321c4f8be6e01e4b",
                "descriptions": 24,
                "models": 223,
                "license": "unspecified",
                "accepted": 0,
                "reason": "no_reuse_license_published_or_embedded",
            },
            {
                "source": "zenodo:15857589",
                "accessibility": "downloaded_md5_verified",
                "artifact": "pmo-dataset.zip",
                "verified_md5": "414d9fbc5aa04bc70db8bf1673f4c69a",
                "candidate_pairs": 55,
                "license": "unspecified",
                "accepted": 0,
                "reason": "no_reuse_license_published_or_embedded",
            },
            {"source": "MaD_via_2025_study", "accessibility": "not_publicly_downloadable", "license": "not_verified", "accepted": 0, "reason": "available_from_corresponding_author_on_request"},
        ],
        "totals": {
            "candidate_pairs": report["candidate_pairs"],
            "accepted": report["accepted"],
            "rejected": report["rejected"],
        },
    }
    (output / "public_import_report.json").write_text(
        json.dumps(source_report, indent=2), encoding="utf-8"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import licensed public BPMN pairs")
    parser.add_argument("--camunda-root", type=Path, required=True)
    parser.add_argument("--accept-camunda-research-only-license", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/public"))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    cases, report = import_camunda(
        args.camunda_root,
        accept_research_only_license=args.accept_camunda_research_only_license,
    )
    write_import(args.output, cases, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
