"""Opt-in text API integration with mocked extraction and real Node layout."""

import json
import os
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from app.bpmn import layout as layout_module
from app.domain import ProcessModel
from app.main import create_app

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = [
    pytest.mark.layout_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_BPMN_LAYOUT_INTEGRATION") != "1",
        reason="set RUN_BPMN_LAYOUT_INTEGRATION=1 to run the local Node worker",
    ),
]


def test_text_api_with_mocked_extraction_and_real_layout() -> None:
    data = json.loads((FIXTURES / "linear_process.json").read_text(encoding="utf-8"))
    model = ProcessModel.model_validate(data)
    client = TestClient(create_app())

    with patch(
        "app.api.v1.routes.text_generation.extract_process", return_value=model
    ):
        response = client.post(
            "/api/v1/generate-from-text",
            json={"description": "An employee reviews a request."},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["review_required"] is True
    root = ElementTree.fromstring(result["bpmn_xml"])
    shapes, edges, subprocesses = layout_module._expected_di(root)
    layout_module._validate_di(root, shapes, edges, subprocesses)
