"""Safe subprocess adapter for deterministic BPMN DI layout."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.bpmn.namespaces import BPMN_NAMESPACE

_BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
_DC = "http://www.omg.org/spec/DD/20100524/DC"
_DI = "http://www.omg.org/spec/DD/20100524/DI"
_DEFAULT_MAX_INPUT_BYTES = 5 * 1024 * 1024
_WORKER_PATH = Path(__file__).with_name("layout_worker") / "worker.mjs"

_SHAPE_TAGS = {
    "startEvent",
    "endEvent",
    "task",
    "subProcess",
    "exclusiveGateway",
    "parallelGateway",
    "participant",
    "lane",
}
_EDGE_TAGS = {"sequenceFlow", "messageFlow"}


class BpmnLayoutError(RuntimeError):
    """Actionable failure raised by the BPMN layout boundary."""

    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _SemanticElement:
    tag: str
    element_id: str
    owner: str | None
    attributes: tuple[tuple[str, str], ...]
    text: str | None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xml(xml: str, *, code: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(xml)
    except (ElementTree.ParseError, ValueError) as error:
        raise BpmnLayoutError(code, f"BPMN XML could not be parsed: {error}.") from error


def _semantic_elements(root: ElementTree.Element) -> tuple[_SemanticElement, ...]:
    records: list[_SemanticElement] = []

    def visit(element: ElementTree.Element, owner: str | None) -> None:
        namespace = element.tag[1:].split("}", 1)[0] if element.tag.startswith("{") else ""
        tag = _local_name(element.tag)
        next_owner = owner
        element_id = element.get("id")
        if namespace == BPMN_NAMESPACE:
            if tag in {"process", "subProcess", "collaboration"} and element_id:
                next_owner = element_id
            if tag not in {"definitions"}:
                attributes = tuple(sorted((key, value) for key, value in element.attrib.items()))
                text = (element.text or "").strip() or None
                records.append(
                    _SemanticElement(tag, element_id or "", owner, attributes, text)
                )
        for child in element:
            visit(child, next_owner)

    visit(root, None)
    return tuple(sorted(records, key=lambda item: (item.tag, item.element_id, item.owner or "", item.attributes, item.text or "")))


def _element_ids(root: ElementTree.Element) -> set[str]:
    return {
        element_id
        for element in root.iter()
        if (element_id := element.get("id")) is not None
        and not element.tag.startswith(f"{{{_BPMNDI}}}")
    }


def _expected_di(root: ElementTree.Element) -> tuple[set[str], set[str], set[str]]:
    shapes: set[str] = set()
    edges: set[str] = set()
    subprocesses: set[str] = set()
    for element in root.iter():
        if not element.tag.startswith(f"{{{BPMN_NAMESPACE}}}"):
            continue
        tag = _local_name(element.tag)
        element_id = element.get("id")
        if element_id and tag in _SHAPE_TAGS:
            shapes.add(element_id)
            if tag == "subProcess":
                subprocesses.add(element_id)
        elif element_id and tag in _EDGE_TAGS:
            edges.add(element_id)
    return shapes, edges, subprocesses


def _reject_expanded_subprocesses(
    root: ElementTree.Element, subprocess_ids: set[str]
) -> None:
    for shape in root.iter(f"{{{_BPMNDI}}}BPMNShape"):
        if (
            shape.get("bpmnElement") in subprocess_ids
            and shape.get("isExpanded", "false").lower() == "true"
        ):
            raise BpmnLayoutError(
                "expanded_subprocess_unsupported",
                "Expanded subprocess layout is not supported; provide collapsed subprocesses only.",
                details={"subprocess_id": shape.get("bpmnElement")},
            )


def _finite(value: str | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except ValueError:
        return False


def _validate_di(
    root: ElementTree.Element,
    expected_shapes: set[str],
    expected_edges: set[str],
    subprocess_ids: set[str],
) -> None:
    real_ids = _element_ids(root)
    found_shapes: set[str] = set()
    found_edges: set[str] = set()
    errors: list[str] = []
    diagrams = list(root.iter(f"{{{_BPMNDI}}}BPMNDiagram"))
    planes = list(root.iter(f"{{{_BPMNDI}}}BPMNPlane"))
    if not diagrams or not planes:
        errors.append("BPMNDiagram and BPMNPlane are required")

    for plane in planes:
        reference = plane.get("bpmnElement")
        if not reference or reference not in real_ids:
            errors.append(f"BPMNPlane references unknown element '{reference}'")

    for shape in root.iter(f"{{{_BPMNDI}}}BPMNShape"):
        reference = shape.get("bpmnElement")
        if not reference or reference not in real_ids:
            errors.append(f"BPMNShape references unknown element '{reference}'")
            continue
        found_shapes.add(reference)
        bounds = shape.find(f"{{{_DC}}}Bounds")
        if bounds is None or not all(
            _finite(bounds.get(name)) for name in ("x", "y", "width", "height")
        ):
            errors.append(f"BPMNShape '{reference}' has non-finite bounds")
        elif float(bounds.get("width", "0")) <= 0 or float(
            bounds.get("height", "0")
        ) <= 0:
            errors.append(f"BPMNShape '{reference}' has non-positive bounds")
        if reference in subprocess_ids and shape.get("isExpanded", "false").lower() == "true":
            errors.append(f"Subprocess '{reference}' is expanded")

    for edge in root.iter(f"{{{_BPMNDI}}}BPMNEdge"):
        reference = edge.get("bpmnElement")
        if not reference or reference not in real_ids:
            errors.append(f"BPMNEdge references unknown element '{reference}'")
            continue
        found_edges.add(reference)
        waypoints = edge.findall(f"{{{_DI}}}waypoint")
        if len(waypoints) < 2 or not all(
            _finite(point.get(axis)) for point in waypoints for axis in ("x", "y")
        ):
            errors.append(f"BPMNEdge '{reference}' has invalid waypoints")

    missing_shapes = sorted(expected_shapes - found_shapes)
    missing_edges = sorted(expected_edges - found_edges)
    if missing_shapes or missing_edges or errors:
        raise BpmnLayoutError(
            "invalid_di",
            "Layout output contains incomplete or invalid BPMN DI.",
            details={
                "missing_shapes": missing_shapes,
                "missing_edges": missing_edges,
                "errors": errors,
            },
        )


def _run_worker(xml: str, timeout_seconds: float) -> dict[str, Any]:
    command = ["node", str(_WORKER_PATH)]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,
        )
    except OSError as error:
        raise BpmnLayoutError(
            "worker_unavailable",
            f"The Node.js layout worker could not be started: {error}.",
        ) from error

    try:
        stdout, stderr = process.communicate(xml, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise BpmnLayoutError(
            "layout_timeout",
            f"BPMN layout exceeded the {timeout_seconds:g}-second timeout.",
            details={"timeout_seconds": timeout_seconds},
        ) from error

    if process.returncode != 0:
        raise BpmnLayoutError(
            "worker_failed",
            "The BPMN layout worker exited with a nonzero status.",
            details={"returncode": process.returncode, "stderr": stderr.strip()},
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BpmnLayoutError(
            "malformed_worker_output",
            "The BPMN layout worker did not return valid JSON.",
            details={"stderr": stderr.strip()},
        ) from error
    if not isinstance(payload, dict):
        raise BpmnLayoutError(
            "malformed_worker_output", "The worker JSON result must be an object."
        )
    return payload


def layout_bpmn(
    xml: str,
    timeout_seconds: float = 10.0,
    *,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
) -> str:
    """Add complete collapsed-subprocess BPMN DI using an isolated Node worker."""
    if timeout_seconds <= 0:
        raise BpmnLayoutError("invalid_timeout", "timeout_seconds must be positive.")
    input_size = len(xml.encode("utf-8"))
    if input_size > max_input_bytes:
        raise BpmnLayoutError(
            "input_too_large",
            f"BPMN XML is {input_size} bytes; the limit is {max_input_bytes} bytes.",
            details={"input_bytes": input_size, "max_input_bytes": max_input_bytes},
        )

    original_root = _parse_xml(xml, code="invalid_input_xml")
    expected_shapes, expected_edges, subprocess_ids = _expected_di(original_root)
    _reject_expanded_subprocesses(original_root, subprocess_ids)
    original_semantics = _semantic_elements(original_root)

    payload = _run_worker(xml, timeout_seconds)
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        raise BpmnLayoutError(
            "malformed_worker_output", "The worker result must contain a warnings list."
        )
    if warnings:
        raise BpmnLayoutError(
            "layout_warnings",
            "The layout engine returned warnings; output was rejected.",
            details={"warnings": warnings},
        )
    output_xml = payload.get("xml")
    if not isinstance(output_xml, str) or not output_xml:
        raise BpmnLayoutError(
            "malformed_worker_output", "The worker result must contain non-empty XML."
        )
    output_root = _parse_xml(output_xml, code="invalid_output_xml")
    if _semantic_elements(output_root) != original_semantics:
        raise BpmnLayoutError(
            "semantic_change",
            "Layout output changed BPMN semantic elements, ownership, attributes, or references.",
        )
    _validate_di(
        output_root, expected_shapes, expected_edges, subprocess_ids
    )
    return output_xml
