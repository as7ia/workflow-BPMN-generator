"""Fast fail-closed tests for the BPMN layout subprocess boundary."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest

from app.bpmn import BpmnLayoutError, layout_bpmn, serialize_bpmn
from app.bpmn import layout as layout_module
from app.bpmn.namespaces import BPMN_NAMESPACE
from app.domain import ProcessModel

FIXTURES = Path(__file__).parent / "fixtures"
BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
DC = "http://www.omg.org/spec/DD/20100524/DC"
DI = "http://www.omg.org/spec/DD/20100524/DI"


def _semantic_xml(name: str = "linear_process.json") -> str:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return serialize_bpmn(ProcessModel.model_validate(data))


def _complete_di(xml: str) -> str:
    root = ElementTree.fromstring(xml)
    shapes, edges, _ = layout_module._expected_di(root)
    collaboration = root.find(f"{{{BPMN_NAMESPACE}}}collaboration")
    assert collaboration is not None
    diagram = ElementTree.SubElement(root, f"{{{BPMNDI}}}BPMNDiagram", {"id": "Diagram_1"})
    plane = ElementTree.SubElement(
        diagram,
        f"{{{BPMNDI}}}BPMNPlane",
        {"id": "Plane_1", "bpmnElement": collaboration.attrib["id"]},
    )
    for index, element_id in enumerate(sorted(shapes)):
        shape = ElementTree.SubElement(
            plane,
            f"{{{BPMNDI}}}BPMNShape",
            {"id": f"Shape_{element_id}", "bpmnElement": element_id},
        )
        ElementTree.SubElement(
            shape,
            f"{{{DC}}}Bounds",
            {"x": str(index * 120), "y": "20", "width": "100", "height": "80"},
        )
    for index, element_id in enumerate(sorted(edges)):
        edge = ElementTree.SubElement(
            plane,
            f"{{{BPMNDI}}}BPMNEdge",
            {"id": f"Edge_{element_id}", "bpmnElement": element_id},
        )
        ElementTree.SubElement(edge, f"{{{DI}}}waypoint", {"x": str(index), "y": "0"})
        ElementTree.SubElement(edge, f"{{{DI}}}waypoint", {"x": str(index + 1), "y": "1"})
    return ElementTree.tostring(root, encoding="unicode")


def _payload(xml: str, warnings: list[object] | None = None) -> dict[str, object]:
    return {"xml": xml, "warnings": warnings or []}


def test_layout_success_returns_validated_worker_xml() -> None:
    source = _semantic_xml()
    expected = _complete_di(source)
    with patch.object(layout_module, "_run_worker", return_value=_payload(expected)):
        assert layout_bpmn(source) == expected


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"xml": "<broken", "warnings": []}, "invalid_output_xml"),
        ({"xml": "<definitions />", "warnings": []}, "semantic_change"),
        ({"xml": "anything", "warnings": [{"message": "partial"}]}, "layout_warnings"),
        ({"warnings": []}, "malformed_worker_output"),
    ],
)
def test_worker_results_fail_closed(payload: dict[str, object], code: str) -> None:
    with patch.object(layout_module, "_run_worker", return_value=payload):
        with pytest.raises(BpmnLayoutError) as captured:
            layout_bpmn(_semantic_xml())
    assert captured.value.code == code


def test_missing_di_is_rejected() -> None:
    source = _semantic_xml()
    with patch.object(layout_module, "_run_worker", return_value=_payload(source)):
        with pytest.raises(BpmnLayoutError) as captured:
            layout_bpmn(source)
    assert captured.value.code == "invalid_di"
    assert captured.value.details["missing_shapes"]
    assert captured.value.details["missing_edges"]


def test_semantic_attribute_change_is_rejected() -> None:
    source = _semantic_xml()
    changed = _complete_di(source).replace('name="Review request"', 'name="Changed"')
    with patch.object(layout_module, "_run_worker", return_value=_payload(changed)):
        with pytest.raises(BpmnLayoutError) as captured:
            layout_bpmn(source)
    assert captured.value.code == "semantic_change"


def test_expanded_subprocess_is_rejected_before_worker() -> None:
    source = _complete_di(_semantic_xml("nested_subprocess.json"))
    root = ElementTree.fromstring(source)
    shape = next(
        item
        for item in root.iter(f"{{{BPMNDI}}}BPMNShape")
        if item.get("bpmnElement") == "verification"
    )
    shape.set("isExpanded", "true")
    expanded = ElementTree.tostring(root, encoding="unicode")
    with patch.object(layout_module, "_run_worker") as worker:
        with pytest.raises(BpmnLayoutError) as captured:
            layout_bpmn(expanded)
    assert captured.value.code == "expanded_subprocess_unsupported"
    worker.assert_not_called()


def test_timeout_kills_and_reaps_worker() -> None:
    class TimedOutProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.calls = 0

        def communicate(self, *_args: object, **_kwargs: object) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["node"], 0.01)
            return "", ""

        def kill(self) -> None:
            self.killed = True

    process = TimedOutProcess()
    with patch.object(subprocess, "Popen", return_value=process):
        with pytest.raises(BpmnLayoutError) as captured:
            layout_module._run_worker("<xml />", 0.01)
    assert captured.value.code == "layout_timeout"
    assert process.killed
    assert process.calls == 2


def test_nonzero_worker_exit_exposes_stderr() -> None:
    class FailedProcess:
        returncode = 7

        def communicate(self, *_args: object, **_kwargs: object) -> tuple[str, str]:
            return "", "layout failed"

    with patch.object(subprocess, "Popen", return_value=FailedProcess()):
        with pytest.raises(BpmnLayoutError) as captured:
            layout_module._run_worker("<xml />", 1)
    assert captured.value.code == "worker_failed"
    assert captured.value.details == {"returncode": 7, "stderr": "layout failed"}


def test_input_size_and_timeout_are_validated_without_starting_worker() -> None:
    with patch.object(layout_module, "_run_worker") as worker:
        with pytest.raises(BpmnLayoutError, match="limit") as size_error:
            layout_bpmn(_semantic_xml(), max_input_bytes=10)
        with pytest.raises(BpmnLayoutError) as timeout_error:
            layout_bpmn(_semantic_xml(), timeout_seconds=0)
    assert size_error.value.code == "input_too_large"
    assert timeout_error.value.code == "invalid_timeout"
    worker.assert_not_called()
