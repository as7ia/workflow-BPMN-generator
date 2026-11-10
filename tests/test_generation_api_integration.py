"""Opt-in end-to-end API tests using the real local Node worker."""

import json
import os
from pathlib import Path
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from app.bpmn import layout as layout_module
from app.main import create_app

FIXTURES = Path(__file__).parent / "fixtures"
CASES = [
    "linear_process.json",
    "exclusive_gateway.json",
    "parallel_split_join.json",
    "loop_corrections.json",
    "two_participants_message.json",
    "nested_subprocess.json",
]

pytestmark = [
    pytest.mark.layout_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_BPMN_LAYOUT_INTEGRATION") != "1",
        reason="set RUN_BPMN_LAYOUT_INTEGRATION=1 to run the local Node worker",
    ),
]


@pytest.mark.parametrize("fixture_name", CASES)
def test_json_to_bpmn_api_with_real_layout(fixture_name: str) -> None:
    client = TestClient(create_app())
    payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    response = client.post("/api/v1/generate-from-json", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    root = ElementTree.fromstring(result["bpmn_xml"])
    shapes, edges, subprocesses = layout_module._expected_di(root)
    layout_module._validate_di(root, shapes, edges, subprocesses)
