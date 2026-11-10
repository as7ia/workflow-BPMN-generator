"""Opt-in integration tests for the pinned local Node layout worker."""

import json
import os
from pathlib import Path

import pytest

from app.bpmn import layout_bpmn, serialize_bpmn
from app.domain import ProcessModel

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SCENARIOS = ROOT / "experiments" / "bpmn-auto-layout-feasibility" / "scenarios"
CASES = [
    FIXTURES / "linear_process.json",
    FIXTURES / "exclusive_gateway.json",
    FIXTURES / "parallel_split_join.json",
    FIXTURES / "loop_corrections.json",
    FIXTURES / "two_participants_message.json",
    FIXTURES / "nested_subprocess.json",
    SCENARIOS / "complex_approval.json",
    SCENARIOS / "complex_collaboration.json",
    SCENARIOS / "complex_nested_process.json",
]

pytestmark = [
    pytest.mark.layout_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_BPMN_LAYOUT_INTEGRATION") != "1",
        reason="set RUN_BPMN_LAYOUT_INTEGRATION=1 to run the local Node worker",
    ),
]


@pytest.mark.parametrize("fixture_path", CASES, ids=lambda path: path.stem)
def test_real_layout_is_complete_for_supported_scenarios(fixture_path: Path) -> None:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    semantic_xml = serialize_bpmn(ProcessModel.model_validate(data))

    assert layout_bpmn(semantic_xml)


def test_real_layout_is_deterministic_for_linear_fixture() -> None:
    data = json.loads((FIXTURES / "linear_process.json").read_text(encoding="utf-8"))
    semantic_xml = serialize_bpmn(ProcessModel.model_validate(data))

    assert layout_bpmn(semantic_xml) == layout_bpmn(semantic_xml)
