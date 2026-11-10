# Semantic BPMN XML serializer

`app.bpmn.serialize_bpmn(model)` validates a `ProcessModel` and serializes the supported schema 1.0 concepts as deterministic, namespace-aware BPMN 2.0 model XML.

The serializer emits definitions, one process per participant, collaborations, participants, lanes, start and end events, tasks, subprocesses, exclusive and parallel gateways, sequence flows, and message flows. Nested scopes are contained by their owning subprocess. Identifiers and references are preserved.

`SequenceFlow.condition` remains descriptive English text. It is emitted as the sequence flow's `name` and never as `conditionExpression`. Message-flow labels are also emitted as names.

Validation levels are distinct:

- XML parsing proves that the output is well formed.
- Domain validation plus serializer checks prove the supported structural invariants and reference mapping.
- Official BPMN XSD conformance is not claimed because this project does not yet validate against the official schemas.
- BPMN DI visual completeness is not present: no diagram, shapes, edges, coordinates, or waypoints are generated.
- Business semantic fidelity to future English input is outside serialization and is not proven here.

A lane used by nodes in more than one process scope cannot be represented once in multiple BPMN lane sets. Such a validated model raises `BpmnSerializationError` instead of duplicating the lane ID or dropping membership.
