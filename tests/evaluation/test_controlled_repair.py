"""Tests for conservative deterministic repair rules."""

import json
from pathlib import Path

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph
from evaluation.controlled_repair import apply_safe_deterministic_repairs

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_unique_unconditional_gateway_flow_repairs_invalid_default() -> None:
    candidate = _fixture("exclusive_gateway.json")
    candidate["nodes"][1]["default_flow_id"] = "missing_flow"

    repaired, report = apply_safe_deterministic_repairs(candidate)
    model = ProcessModel.model_validate(repaired)
    validate_process_graph(model)

    assert repaired["nodes"][1]["default_flow_id"] == "flow_reject"
    assert report["changes"] == [{
        "code": "resolve_unique_gateway_default_flow",
        "location": ["nodes", 1, "default_flow_id"],
        "resolved_id": "flow_reject",
    }]


def test_singleton_existing_reference_object_is_unwrapped() -> None:
    candidate = _fixture("exclusive_gateway.json")
    candidate["sequence_flows"][0]["target_ref"] = {"id": "decision"}

    repaired, report = apply_safe_deterministic_repairs(candidate)
    model = ProcessModel.model_validate(repaired)
    validate_process_graph(model)

    assert repaired["sequence_flows"][0]["target_ref"] == "decision"
    assert report["changes"][0]["code"] == "unwrap_unique_existing_reference"

    ambiguous = _fixture("exclusive_gateway.json")
    ambiguous["sequence_flows"][0]["target_ref"] = ["decision", "approve"]
    unchanged, ambiguous_report = apply_safe_deterministic_repairs(ambiguous)
    assert unchanged["sequence_flows"][0]["target_ref"] == ["decision", "approve"]
    assert ambiguous_report["refusals"][0]["code"] == (
        "reference_not_uniquely_resolvable"
    )


def test_null_forbidden_field_is_removed_but_non_null_value_is_preserved() -> None:
    null_candidate = _fixture("exclusive_gateway.json")
    null_candidate["lanes"] = [{
        "id": "operations", "name": "Operations",
        "participant_id": "company", "lane_id": None,
    }]
    repaired, report = apply_safe_deterministic_repairs(null_candidate)
    validate_process_graph(ProcessModel.model_validate(repaired))
    assert "lane_id" not in repaired["lanes"][0]
    assert report["changes"][0]["code"] == "remove_null_forbidden_field"

    meaningful_candidate = _fixture("exclusive_gateway.json")
    meaningful_candidate["nodes"][2]["parent_lane_id"] = "management"
    refused, refusal_report = apply_safe_deterministic_repairs(meaningful_candidate)
    assert refused["nodes"][2]["parent_lane_id"] == "management"
    assert refusal_report["refusals"][0]["code"] == (
        "non_null_forbidden_field_requires_semantic_review"
    )


def test_disconnected_graph_is_not_connected_by_guessing() -> None:
    candidate = _fixture("parallel_split_join.json")
    candidate["sequence_flows"] = [
        flow for flow in candidate["sequence_flows"] if flow["id"] != "f2"
    ]

    repaired, report = apply_safe_deterministic_repairs(candidate)

    assert repaired == candidate
    assert report["changes"] == []
    model = ProcessModel.model_validate(repaired)
    try:
        validate_process_graph(model)
    except ProcessGraphValidationError as error:
        assert any(issue.code == "unreachable_from_start" for issue in error.issues)
    else:
        raise AssertionError("Disconnected graph was unexpectedly accepted")
