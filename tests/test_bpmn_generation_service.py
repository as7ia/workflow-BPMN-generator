"""Fast tests for framework-independent BPMN orchestration."""

import json
from pathlib import Path
from unittest.mock import call, patch

from app.domain import ProcessModel
from app.services.bpmn_generation import generate_bpmn

FIXTURES = Path(__file__).parent / "fixtures"


def test_service_preserves_input_and_runs_pipeline_in_order() -> None:
    data = json.loads((FIXTURES / "linear_process.json").read_text(encoding="utf-8"))
    model = ProcessModel.model_validate(data)
    before = model.model_dump(mode="json")
    order = []

    with (
        patch(
            "app.services.bpmn_generation.validate_process_graph",
            side_effect=lambda value: order.append(("validate", value)),
        ) as validate,
        patch(
            "app.services.bpmn_generation.serialize_bpmn",
            side_effect=lambda value: order.append(("serialize", value)) or "semantic",
        ) as serialize,
        patch(
            "app.services.bpmn_generation.layout_bpmn",
            side_effect=lambda xml, **kwargs: order.append(("layout", xml, kwargs)) or "layouted",
        ) as layout,
    ):
        result = generate_bpmn(model, timeout_seconds=3.5)

    assert result == "layouted"
    assert model.model_dump(mode="json") == before
    assert order == [
        ("validate", model),
        ("serialize", model),
        ("layout", "semantic", {"timeout_seconds": 3.5}),
    ]
    assert validate.call_args == call(model)
    assert serialize.call_args == call(model)
    assert layout.call_args == call("semantic", timeout_seconds=3.5)
