"""Deterministic serialization of validated process models to BPMN 2.0 XML."""

from collections import defaultdict
from xml.etree import ElementTree

from app.bpmn.namespaces import BPMN_NAMESPACE, bpmn_tag
from app.domain import ProcessModel, validate_process_graph
from app.domain.process_model import (
    Lane,
    NodeType,
    ProcessNode,
    ProcessScope,
    SequenceFlow,
)


class BpmnSerializationError(ValueError):
    """Raised when a valid domain model cannot be represented without loss."""


class _Serializer:
    """Stateful serializer used for one deterministic serialization operation."""

    NODE_TAGS = {
        NodeType.START_EVENT: "startEvent",
        NodeType.END_EVENT: "endEvent",
        NodeType.TASK: "task",
        NodeType.SUBPROCESS: "subProcess",
        NodeType.EXCLUSIVE_GATEWAY: "exclusiveGateway",
        NodeType.PARALLEL_GATEWAY: "parallelGateway",
    }

    def __init__(self, model: ProcessModel) -> None:
        self.model = model
        self.nodes_by_scope: dict[str, list[ProcessNode]] = defaultdict(list)
        self.flows_by_scope: dict[str, list[SequenceFlow]] = defaultdict(list)
        self.child_scope_by_subprocess: dict[str, ProcessScope] = {}
        self.root_scope_by_participant: dict[str, ProcessScope] = {}
        self.lanes_by_scope: dict[str, list[Lane]] = defaultdict(list)
        self.serialized_scopes: set[str] = set()
        self.active_scopes: set[str] = set()
        self.used_ids = self._domain_ids()
        self._index_model()

    def _domain_ids(self) -> set[str]:
        ids = {self.model.id}
        for collection in (
            self.model.participants,
            self.model.lanes,
            self.model.scopes,
            self.model.nodes,
            self.model.sequence_flows,
            self.model.message_flows,
        ):
            ids.update(item.id for item in collection)
        return ids

    def _allocate_id(self, base: str) -> str:
        candidate = base
        suffix = 2
        while candidate in self.used_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.used_ids.add(candidate)
        return candidate

    def _index_model(self) -> None:
        node_by_id = {node.id: node for node in self.model.nodes}
        for node in self.model.nodes:
            self.nodes_by_scope[node.scope_id].append(node)
        for flow in self.model.sequence_flows:
            source = node_by_id[flow.source_ref]
            self.flows_by_scope[source.scope_id].append(flow)
        for scope in self.model.scopes:
            if scope.parent_scope_id is None:
                self.root_scope_by_participant[scope.participant_id] = scope
            elif scope.subprocess_node_id is not None:
                self.child_scope_by_subprocess[scope.subprocess_node_id] = scope
        self._index_lanes()

    def _index_lanes(self) -> None:
        node_scopes_by_lane: dict[str, set[str]] = defaultdict(set)
        for node in self.model.nodes:
            if node.lane_id is not None:
                node_scopes_by_lane[node.lane_id].add(node.scope_id)

        for lane in self.model.lanes:
            assigned_scopes = node_scopes_by_lane.get(lane.id, set())
            if len(assigned_scopes) > 1:
                scope_names = ", ".join(sorted(assigned_scopes))
                raise BpmnSerializationError(
                    f"Lane '{lane.id}' is assigned to nodes in multiple scopes: "
                    f"{scope_names}. Schema 1.0 cannot serialize one lane into "
                    "multiple BPMN lane sets without duplicating its ID."
                )
            if assigned_scopes:
                scope_id = next(iter(assigned_scopes))
            else:
                scope_id = self.root_scope_by_participant[lane.participant_id].id
            self.lanes_by_scope[scope_id].append(lane)

    def serialize(self) -> str:
        definitions = ElementTree.Element(
            bpmn_tag("definitions"),
            {
                "id": self._allocate_id(f"Definitions_{self.model.id}"),
                "name": self.model.name,
                "targetNamespace": f"urn:workflow-bpmn-generator:{self.model.id}",
            },
        )

        for participant in self.model.participants:
            root_scope = self.root_scope_by_participant[participant.id]
            process = ElementTree.SubElement(
                definitions,
                bpmn_tag("process"),
                {"id": root_scope.id, "name": participant.name},
            )
            self._serialize_scope(root_scope, process)

        collaboration = ElementTree.SubElement(
            definitions,
            bpmn_tag("collaboration"),
            {"id": self._allocate_id(f"Collaboration_{self.model.id}")},
        )
        for participant in self.model.participants:
            root_scope = self.root_scope_by_participant[participant.id]
            ElementTree.SubElement(
                collaboration,
                bpmn_tag("participant"),
                {
                    "id": participant.id,
                    "name": participant.name,
                    "processRef": root_scope.id,
                },
            )
        for flow in self.model.message_flows:
            attributes = {
                "id": flow.id,
                "sourceRef": flow.source_ref,
                "targetRef": flow.target_ref,
            }
            if flow.label is not None:
                attributes["name"] = flow.label
            ElementTree.SubElement(
                collaboration, bpmn_tag("messageFlow"), attributes
            )

        if self.serialized_scopes != {scope.id for scope in self.model.scopes}:
            missing = sorted(
                {scope.id for scope in self.model.scopes} - self.serialized_scopes
            )
            raise BpmnSerializationError(
                f"Process scopes were not serialized: {', '.join(missing)}."
            )

        ElementTree.indent(definitions, space="  ")
        return ElementTree.tostring(
            definitions, encoding="unicode", xml_declaration=True
        )

    def _serialize_scope(
        self, scope: ProcessScope, container: ElementTree.Element
    ) -> None:
        if scope.id in self.active_scopes:
            raise BpmnSerializationError(
                f"Recursive subprocess scope '{scope.id}' cannot be serialized."
            )
        if scope.id in self.serialized_scopes:
            raise BpmnSerializationError(
                f"Process scope '{scope.id}' would be serialized more than once."
            )
        self.active_scopes.add(scope.id)

        self._serialize_lanes(scope, container)
        for node in self.nodes_by_scope[scope.id]:
            attributes = {"id": node.id, "name": node.name}
            if node.default_flow_id is not None:
                attributes["default"] = node.default_flow_id
            node_element = ElementTree.SubElement(
                container, bpmn_tag(self.NODE_TAGS[node.type]), attributes
            )
            if node.type == NodeType.SUBPROCESS:
                child_scope = self.child_scope_by_subprocess.get(node.id)
                if child_scope is None:
                    raise BpmnSerializationError(
                        f"Subprocess node '{node.id}' has no nested scope."
                    )
                self._serialize_scope(child_scope, node_element)

        for flow in self.flows_by_scope[scope.id]:
            attributes = {
                "id": flow.id,
                "sourceRef": flow.source_ref,
                "targetRef": flow.target_ref,
            }
            if flow.condition is not None:
                attributes["name"] = flow.condition
            ElementTree.SubElement(
                container, bpmn_tag("sequenceFlow"), attributes
            )

        self.active_scopes.remove(scope.id)
        self.serialized_scopes.add(scope.id)

    def _serialize_lanes(
        self, scope: ProcessScope, container: ElementTree.Element
    ) -> None:
        lanes = self.lanes_by_scope.get(scope.id, [])
        if not lanes:
            return
        lane_set = ElementTree.SubElement(
            container,
            bpmn_tag("laneSet"),
            {"id": self._allocate_id(f"LaneSet_{scope.id}")},
        )
        for lane in lanes:
            lane_element = ElementTree.SubElement(
                lane_set,
                bpmn_tag("lane"),
                {"id": lane.id, "name": lane.name},
            )
            for node in self.nodes_by_scope[scope.id]:
                if node.lane_id == lane.id:
                    reference = ElementTree.SubElement(
                        lane_element, bpmn_tag("flowNodeRef")
                    )
                    reference.text = node.id


def serialize_bpmn(model: ProcessModel) -> str:
    """Validate and deterministically serialize a process model to BPMN XML.

    The output contains semantic BPMN model elements only. It does not include
    BPMN DI shapes, edges, waypoints, or any executable condition expressions.
    """
    validate_process_graph(model)
    return _Serializer(model).serialize()
