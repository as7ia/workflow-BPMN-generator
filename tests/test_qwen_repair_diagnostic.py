"""Mock-only tests for the explicit Qwen repair diagnostic."""

import json
from pathlib import Path

import pytest

from app.domain import ProcessModel
from app.llm import OllamaSettings
from experiments.qwen_repair_diagnostic import (
    capture_live_diagnostic,
    compare_candidates,
    render_report,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeTransport:
    def __init__(self, candidates: list[dict[str, object]]) -> None:
        self.responses = [
            {
                "message": {
                    "content": json.dumps(candidate),
                }
            }
            for candidate in candidates
        ]
        self.calls: list[dict[str, object]] = []

    def post_json(self, url, payload, *, timeout_seconds, max_response_bytes):
        self.calls.append(payload)
        return self.responses[len(self.calls) - 1]


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_live_diagnostic_captures_two_candidates_without_network() -> None:
    initial = _fixture("invalid_dangling_sequence_flow.json")
    repaired = _fixture("linear_process.json")
    transport = FakeTransport([initial, repaired])

    capture = capture_live_diagnostic(
        "An operations employee reviews a request and then completes it.",
        settings=OllamaSettings(),
        transport=transport,
    )

    assert capture.initial_candidate == initial
    assert capture.repaired_candidate == repaired
    assert len(transport.calls) == 2
    assert capture.report["initial_graph_issues"]
    assert capture.report["repaired_graph_issues"] == []
    assert "original_candidate" in transport.calls[1]["messages"][1]["content"]


def test_comparison_reports_collections_and_requested_indexes() -> None:
    initial_data = _fixture("nested_subprocess.json")
    repaired_data = json.loads(json.dumps(initial_data))
    repaired_data["nodes"][5]["name"] = "Changed internal task"
    repaired_data["sequence_flows"].append(
        {
            "id": "flow_extra",
            "source_ref": repaired_data["nodes"][4]["id"],
            "target_ref": repaired_data["nodes"][5]["id"],
        }
    )

    report = compare_candidates(
        "The subprocess Fulfill request comprises the following steps.",
        ProcessModel.model_validate(initial_data),
        ProcessModel.model_validate(repaired_data),
    )

    assert set(report["collections"]) == {
        "participants",
        "scopes",
        "nodes",
        "sequence_flows",
    }
    node_focus = report["focus"]["nodes[5]"]
    scope_focus = report["focus"]["scopes[1]"]
    assert node_focus["before"]["item"] != node_focus["after"]["item"]
    assert node_focus["before"]["incoming_flow_ids"]
    assert scope_focus["before"]["item"] == scope_focus["after"]["item"]
    assert "flow_extra" not in scope_focus["before"]["internal_flow_ids"]
    assert "flow_extra" in scope_focus["after"]["internal_flow_ids"]
    assert scope_focus["before"]["start_event_ids"]
    assert scope_focus["before"]["end_event_ids"]
    assert report["collections"]["sequence_flows"]["added"] == ["flow_extra"]


def test_subprocess_added_by_repair_is_flagged_and_full_candidates_are_not_rendered() -> None:
    initial_data = _fixture("linear_process.json")
    repaired_data = json.loads(json.dumps(initial_data))
    repaired_data["nodes"].append(
        {
            "id": "node_unrequested_subprocess",
            "type": "subprocess",
            "name": "Unrequested subprocess",
            "participant_id": repaired_data["participants"][0]["id"],
            "lane_id": None,
            "scope_id": repaired_data["scopes"][0]["id"],
            "default_flow_id": None,
        }
    )

    report = compare_candidates(
        "An employee reviews and completes a request.",
        ProcessModel.model_validate(initial_data),
        ProcessModel.model_validate(repaired_data),
    )
    finding = report["subprocess_justification"][0]
    rendered = render_report(report)

    assert finding["status"] == "unjustified_added_by_repair"
    assert "node_unrequested_subprocess" in rendered
    assert '"schema_version"' not in rendered
    assert '"message_flows"' not in rendered


def test_valid_initial_candidate_is_not_sent_for_repair() -> None:
    valid = _fixture("linear_process.json")
    transport = FakeTransport([valid])

    with pytest.raises(ValueError, match="nothing to repair"):
        capture_live_diagnostic(
            "An employee reviews a request.",
            settings=OllamaSettings(),
            transport=transport,
        )

    assert len(transport.calls) == 1
