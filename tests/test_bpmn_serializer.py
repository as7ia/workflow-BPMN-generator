"""Structural tests for deterministic semantic BPMN XML serialization."""

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from app.bpmn import BpmnSerializationError, serialize_bpmn
from app.bpmn.namespaces import BPMN_NAMESPACE
from app.domain import ProcessGraphValidationError, ProcessModel

FIXTURES = Path(__file__).parent / "fixtures"
NS = {"bpmn": BPMN_NAMESPACE}
VALID_FIXTURES = [
    "linear_process.json",
    "exclusive_gateway.json",
    "parallel_split_join.json",
    "loop_corrections.json",
    "two_participants_message.json",
    "nested_subprocess.json",
]
NODE_TAGS = {
    "start_event": "startEvent",
    "end_event": "endEvent",
    "task": "task",
    "subprocess": "subProcess",
    "exclusive_gateway": "exclusiveGateway",
    "parallel_gateway": "parallelGateway",
}


def load_data(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def load_model(name: str) -> ProcessModel:
    return ProcessModel.model_validate(load_data(name))


def serialize_root(name: str) -> ElementTree.Element:
    return ElementTree.fromstring(serialize_bpmn(load_model(name)))


@pytest.mark.parametrize("fixture_name", VALID_FIXTURES)
def test_all_valid_fixtures_are_well_formed_and_deterministic(
    fixture_name: str,
) -> None:
    model = load_model(fixture_name)

    first = serialize_bpmn(model)
    second = serialize_bpmn(model)
    root = ElementTree.fromstring(first)

    assert first == second
    assert root.tag == f"{{{BPMN_NAMESPACE}}}definitions"
    assert root.attrib["id"].startswith("Definitions_")
    assert root.attrib["targetNamespace"].endswith(f":{model.id}")


@pytest.mark.parametrize("fixture_name", VALID_FIXTURES)
def test_node_ids_types_labels_and_sequence_references_are_preserved(
    fixture_name: str,
) -> None:
    model = load_model(fixture_name)
    root = ElementTree.fromstring(serialize_bpmn(model))

    for node in model.nodes:
        elements = root.findall(f".//bpmn:{NODE_TAGS[node.type.value]}", NS)
        element = next(item for item in elements if item.attrib.get("id") == node.id)
        assert element.attrib["name"] == node.name

    for flow in model.sequence_flows:
        element = root.find(f".//bpmn:sequenceFlow[@id='{flow.id}']", NS)
        assert element is not None
        assert element.attrib["sourceRef"] == flow.source_ref
        assert element.attrib["targetRef"] == flow.target_ref


def test_exclusive_default_and_descriptive_condition_are_preserved() -> None:
    root = serialize_root("exclusive_gateway.json")

    gateway = root.find(".//bpmn:exclusiveGateway[@id='decision']", NS)
    conditional_flow = root.find(
        ".//bpmn:sequenceFlow[@id='flow_approve']", NS
    )

    assert gateway is not None
    assert gateway.attrib["default"] == "flow_reject"
    assert conditional_flow is not None
    assert conditional_flow.attrib["name"] == "Request is acceptable"
    assert root.find(".//bpmn:conditionExpression", NS) is None


def test_parallel_split_and_join_topology_is_preserved() -> None:
    root = serialize_root("parallel_split_join.json")

    gateways = {
        item.attrib["id"]
        for item in root.findall(".//bpmn:parallelGateway", NS)
    }
    flows = {
        (item.attrib["sourceRef"], item.attrib["targetRef"])
        for item in root.findall(".//bpmn:sequenceFlow", NS)
    }

    assert gateways == {"split", "join"}
    assert {("split", "financial"), ("split", "legal")} <= flows
    assert {("financial", "join"), ("legal", "join")} <= flows


def test_business_loop_back_edge_is_preserved() -> None:
    root = serialize_root("loop_corrections.json")

    back_edge = root.find(".//bpmn:sequenceFlow[@id='f5']", NS)

    assert back_edge is not None
    assert back_edge.attrib == {
        "id": "f5",
        "sourceRef": "correct",
        "targetRef": "review",
    }


def test_nested_scope_is_inside_owning_subprocess() -> None:
    root = serialize_root("nested_subprocess.json")

    subprocess = root.find(".//bpmn:subProcess[@id='verification']", NS)

    assert subprocess is not None
    assert subprocess.find("./bpmn:startEvent[@id='sub_start']", NS) is not None
    assert subprocess.find("./bpmn:task[@id='check']", NS) is not None
    assert subprocess.find("./bpmn:endEvent[@id='sub_end']", NS) is not None
    assert subprocess.find("./bpmn:sequenceFlow[@id='sf1']", NS) is not None
    assert root.find("./bpmn:process/bpmn:task[@id='check']", NS) is None


def test_collaboration_participants_processes_and_message_flow() -> None:
    root = serialize_root("two_participants_message.json")
    collaboration = root.find("./bpmn:collaboration", NS)

    assert collaboration is not None
    participants = {
        item.attrib["id"]: item.attrib["processRef"]
        for item in collaboration.findall("./bpmn:participant", NS)
    }
    assert participants == {
        "customer": "customer_process",
        "supplier": "supplier_process",
    }
    assert {
        process.attrib["id"]
        for process in root.findall("./bpmn:process", NS)
    } == {"customer_process", "supplier_process"}

    message = collaboration.find("./bpmn:messageFlow[@id='order_message']", NS)
    assert message is not None
    assert message.attrib == {
        "id": "order_message",
        "sourceRef": "send_order",
        "targetRef": "receive_order",
        "name": "Purchase order",
    }
    assert root.find("./bpmn:process/bpmn:messageFlow", NS) is None


def test_lane_membership_is_preserved() -> None:
    root = serialize_root("linear_process.json")
    lane = root.find("./bpmn:process/bpmn:laneSet/bpmn:lane", NS)

    assert lane is not None
    assert lane.attrib == {"id": "operations", "name": "Operations"}
    assert [
        reference.text for reference in lane.findall("./bpmn:flowNodeRef", NS)
    ] == ["start", "review", "end"]


def test_invalid_domain_model_is_rejected_before_serialization() -> None:
    model = load_model("invalid_dangling_sequence_flow.json")

    with pytest.raises(ProcessGraphValidationError):
        serialize_bpmn(model)


def test_serialization_does_not_mutate_process_model() -> None:
    model = load_model("nested_subprocess.json")
    before = model.model_dump(mode="json")

    serialize_bpmn(model)

    assert model.model_dump(mode="json") == before


def test_lane_spanning_multiple_scopes_fails_explicitly() -> None:
    data = load_data("nested_subprocess.json")
    data["lanes"] = [
        {"id": "shared_lane", "name": "Shared responsibility", "participant_id": "company"}
    ]
    data["nodes"][0]["lane_id"] = "shared_lane"
    data["nodes"][4]["lane_id"] = "shared_lane"
    model = ProcessModel.model_validate(data)

    with pytest.raises(BpmnSerializationError, match="multiple scopes"):
        serialize_bpmn(model)
