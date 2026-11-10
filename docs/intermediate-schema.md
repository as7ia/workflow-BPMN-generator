# Intermediate Process Schema 1.0

The intermediate model is a typed JSON representation between future English process understanding and future BPMN serialization. It records process meaning without producing BPMN XML or diagram coordinates.

## Parsing and validation

Schema and graph validation are intentionally separate:

```python
from app.domain import ProcessModel, validate_process_graph

model = ProcessModel.model_validate(payload)  # Pydantic shape and type checks
validate_process_graph(model)                 # references and graph invariants
```

Pydantic raises `ValidationError` for malformed fields, unsupported node types, invalid identifiers, unknown fields, or a schema version other than `1.0`. Graph validation raises `ProcessGraphValidationError`; its ordered `issues` contain stable `code`, `location`, and `message` values. Validation never modifies or repairs input.

## Design

The root `ProcessModel` contains:

- `participants`: pools and future BPMN `participant` elements.
- `lanes`: named responsibilities owned by a participant and future BPMN lanes.
- `scopes`: one control-flow graph for a root process or nested subprocess. Schema 1.0 requires exactly one root scope per participant.
- `nodes`: start events, end events, tasks, subprocess placeholders, exclusive gateways, and parallel gateways.
- `sequence_flows`: directed edges within exactly one participant and scope. A back edge represents a business loop; no separate loop flag is needed.
- `message_flows`: communication across participants. Endpoints may be participant IDs or IDs of eligible interaction activities (tasks and subprocesses).

Every node declares `participant_id`, optional `lane_id`, and `scope_id`. This deliberate ownership redundancy makes incorrect cross-pool control flow detectable before serialization.

The future serializer mapping is intentionally direct:

| Schema element | Future BPMN 2.0 element |
| --- | --- |
| Participant | `bpmn:participant` (pool) |
| Lane | `bpmn:lane` |
| Root process scope | `bpmn:process` referenced by a participant |
| Start event | `bpmn:startEvent` |
| End event | `bpmn:endEvent` |
| Task | `bpmn:task` |
| Subprocess node and child scope | `bpmn:subProcess` and its contained flow elements |
| Exclusive gateway | `bpmn:exclusiveGateway` |
| Parallel gateway | `bpmn:parallelGateway` |
| Sequence flow | `bpmn:sequenceFlow` |
| Message flow | `bpmn:messageFlow` in a collaboration |

This table is a design contract only; no serializer exists in Mission 002.

### Nested subprocess representation

A subprocess node remains in its parent's scope. Its contained graph has exactly one separate `ProcessScope` with both `parent_scope_id` and `subprocess_node_id`. The child scope has its own start and end event. Scope parent relationships must be acyclic. Sequence flows do not cross the boundary: the parent graph connects to the subprocess node, and the child graph connects its internal nodes.

`SequenceFlow.condition` is a human-readable English branch label only. It is not executable code and does not carry a formal expression language. A future serializer must preserve it as the descriptive sequence-flow name; it must not emit an executable BPMN condition expression unless a separate formal-expression model is introduced.

### Example

```json
{
  "schema_version": "1.0",
  "id": "approval",
  "name": "Document approval",
  "participants": [{"id": "company", "name": "Company"}],
  "lanes": [],
  "scopes": [{"id": "main", "participant_id": "company"}],
  "nodes": [
    {"id": "start", "type": "start_event", "name": "Document received", "participant_id": "company", "scope_id": "main"},
    {"id": "approve", "type": "task", "name": "Approve document", "participant_id": "company", "scope_id": "main"},
    {"id": "end", "type": "end_event", "name": "Document approved", "participant_id": "company", "scope_id": "main"}
  ],
  "sequence_flows": [
    {"id": "f1", "source_ref": "start", "target_ref": "approve"},
    {"id": "f2", "source_ref": "approve", "target_ref": "end"}
  ],
  "message_flows": []
}
```

Names and labels are required to be non-empty and are an English-input contract. Schema 1.0 does not attempt unreliable automatic language detection.

## Deterministic graph rules

- IDs are unique across the model and all element collections.
- Every participant owns exactly one root process scope.
- Every subprocess node owns exactly one nested scope, and scope parent relationships are acyclic.
- Lane, participant, scope, node, and flow references resolve.
- Node, lane, scope, and participant ownership agrees.
- Sequence flows remain within one participant and one process scope.
- Message flows resolve to supported endpoints and cross participant boundaries. Node endpoints are limited to tasks and subprocesses in schema 1.0; events and gateways are rejected.
- Every scope has at least one start event and one end event.
- Start events have no incoming sequence flow; end events have no outgoing sequence flow.
- Every node is reachable from a start and can reach an end in its scope.
- Traversal is iterative and cycle-safe; valid loops are accepted.
- An exclusive gateway default references one of its outgoing flows. Every other branch has a condition label when the gateway splits, and the default has no condition.
- Parallel gateway outgoing flows cannot carry conditions in schema 1.0.

## Boundaries and limitations

This schema does not implement BPMN execution semantics. It does not yet represent boundary events, intermediate events, event-based or inclusive gateways, call activities, data objects, compensation, transactions, multi-instance behavior, or gateway pairing semantics. A subprocess is a structural placeholder plus a nested graph.

Graph validity only proves the documented structural invariants. It does not prove that an English description was understood faithfully, that branches are logically exhaustive or mutually exclusive, or that parallel synchronization is behaviorally sound. Those are semantic-fidelity concerns for later process-understanding work.

BPMN XML serialization, BPMN DI shapes/edges, layout, and XML schema validation are separate later stages and are not implemented here.

## Run tests

```powershell
python -m pytest tests/test_process_model.py
python -m pytest
```

