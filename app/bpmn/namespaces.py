"""XML namespaces used by semantic BPMN serialization."""

from xml.etree import ElementTree

BPMN_NAMESPACE = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMN_PREFIX = "bpmn"

ElementTree.register_namespace(BPMN_PREFIX, BPMN_NAMESPACE)


def bpmn_tag(local_name: str) -> str:
    """Return an ElementTree qualified name in the BPMN model namespace."""
    return f"{{{BPMN_NAMESPACE}}}{local_name}"
