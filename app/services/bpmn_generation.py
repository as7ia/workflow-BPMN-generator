"""Orchestration service for validated JSON-to-BPMN generation."""

from app.bpmn import layout_bpmn, serialize_bpmn
from app.domain import ProcessModel, validate_process_graph


def generate_bpmn(model: ProcessModel, *, timeout_seconds: float = 10.0) -> str:
    """Validate, serialize, and lay out a process without mutating the model."""
    validate_process_graph(model)
    semantic_xml = serialize_bpmn(model)
    return layout_bpmn(semantic_xml, timeout_seconds=timeout_seconds)
