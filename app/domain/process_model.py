"""Typed intermediate representation for supported BPMN process concepts."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", min_length=1),
]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    """Base model that rejects fields not declared by the schema."""

    model_config = ConfigDict(extra="forbid")


class NodeType(StrEnum):
    """Node kinds supported by the intermediate schema."""

    START_EVENT = "start_event"
    END_EVENT = "end_event"
    TASK = "task"
    SUBPROCESS = "subprocess"
    EXCLUSIVE_GATEWAY = "exclusive_gateway"
    PARALLEL_GATEWAY = "parallel_gateway"


class Participant(StrictModel):
    """A BPMN participant/pool and its future process owner."""

    id: Identifier
    name: Label


class Lane(StrictModel):
    """A responsibility lane contained by one participant."""

    id: Identifier
    name: Label
    participant_id: Identifier


class ProcessScope(StrictModel):
    """A root process or the contents of one nested subprocess.

    Root scopes omit ``parent_scope_id`` and ``subprocess_node_id``. A nested
    scope supplies both and points to the subprocess node that owns it.
    """

    id: Identifier
    participant_id: Identifier
    parent_scope_id: Identifier | None = None
    subprocess_node_id: Identifier | None = None


class ProcessNode(StrictModel):
    """A future BPMN flow node with explicit pool, lane and scope ownership."""

    id: Identifier
    type: NodeType
    name: Label
    participant_id: Identifier
    lane_id: Identifier | None = None
    scope_id: Identifier
    default_flow_id: Identifier | None = None


class SequenceFlow(StrictModel):
    """Directed control flow; cycles/back edges are represented normally.

    ``condition`` is a human-readable English branch label, not an executable
    expression or expression-language payload.
    """

    id: Identifier
    source_ref: Identifier
    target_ref: Identifier
    condition: Label | None = None


class MessageFlow(StrictModel):
    """Communication between participants, using node or pool endpoints."""

    id: Identifier
    source_ref: Identifier
    target_ref: Identifier
    label: Label | None = None


class ProcessModel(StrictModel):
    """Versioned, transport-independent process representation."""

    schema_version: Literal["1.0"]
    id: Identifier
    name: Label
    participants: list[Participant] = Field(min_length=1)
    lanes: list[Lane] = Field(default_factory=list)
    scopes: list[ProcessScope] = Field(min_length=1)
    nodes: list[ProcessNode] = Field(min_length=1)
    sequence_flows: list[SequenceFlow] = Field(default_factory=list)
    message_flows: list[MessageFlow] = Field(default_factory=list)
