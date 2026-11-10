"""Generate semantic BPMN XML inputs for the isolated layout experiment."""

import json
from pathlib import Path

from app.bpmn import serialize_bpmn
from app.domain import ProcessModel

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
OUTPUT = Path(__file__).resolve().parent / "inputs"
VALID_FIXTURES = (
    "linear_process",
    "exclusive_gateway",
    "parallel_split_join",
    "loop_corrections",
    "two_participants_message",
    "nested_subprocess",
)
STRESS_FIXTURES = (
    "complex_approval",
    "complex_collaboration",
    "complex_nested_process",
)
SCENARIOS = Path(__file__).resolve().parent / "scenarios"


def add_expanded_subprocess_seed(xml: str) -> str:
    """Add authored DI whose sole purpose is marking the subprocess expanded."""
    xml = xml.replace(
        "<bpmn:definitions ",
        '<bpmn:definitions xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" '
        'xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" ',
        1,
    )
    seed = """
  <bpmndi:BPMNDiagram id="ExpansionSeedDiagram">
    <bpmndi:BPMNPlane id="ExpansionSeedPlane" bpmnElement="main">
      <bpmndi:BPMNShape id="ExpansionSeedShape" bpmnElement="verification" isExpanded="true">
        <dc:Bounds x="100" y="100" width="300" height="200" />
      </bpmndi:BPMNShape>
    </bpmndi:BPMNPlane>
  </bpmndi:BPMNDiagram>
"""
    return xml.replace("</bpmn:definitions>", f"{seed}</bpmn:definitions>")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for fixture_name in VALID_FIXTURES:
        data = json.loads(
            (FIXTURES / f"{fixture_name}.json").read_text(encoding="utf-8")
        )
        xml = serialize_bpmn(ProcessModel.model_validate(data))
        (OUTPUT / f"{fixture_name}.bpmn").write_text(xml, encoding="utf-8")
        if fixture_name == "nested_subprocess":
            (OUTPUT / "nested_subprocess_expanded.bpmn").write_text(
                add_expanded_subprocess_seed(xml), encoding="utf-8"
            )
    for fixture_name in STRESS_FIXTURES:
        data = json.loads(
            (SCENARIOS / f"{fixture_name}.json").read_text(encoding="utf-8")
        )
        xml = serialize_bpmn(ProcessModel.model_validate(data))
        (OUTPUT / f"{fixture_name}.bpmn").write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    main()
