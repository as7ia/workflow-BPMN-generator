"""Tests for intermediate process schema and deterministic graph validation."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict[str, Any]:
    """Load one immutable-on-disk JSON fixture."""
    with (FIXTURES / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def issue_codes(data: dict[str, Any]) -> set[str]:
    """Parse a schema-valid document and return graph error codes."""
    model = ProcessModel.model_validate(data)
    with pytest.raises(ProcessGraphValidationError) as exc_info:
        validate_process_graph(model)
    return {issue.code for issue in exc_info.value.issues}


@pytest.mark.parametrize(
    "fixture_name",
    [
        "linear_process.json",
        "exclusive_gateway.json",
        "parallel_split_join.json",
        "loop_corrections.json",
        "two_participants_message.json",
        "nested_subprocess.json",
    ],
)
def test_valid_process_fixtures(fixture_name: str) -> None:
    model = ProcessModel.model_validate(load_json(fixture_name))

    validate_process_graph(model)


@pytest.mark.parametrize(
    ("fixture_name", "expected_code"),
    [
        ("invalid_dangling_sequence_flow.json", "unknown_sequence_target"),
        ("invalid_cross_pool_sequence_flow.json", "cross_participant_sequence_flow"),
        ("duplicate_ids.json", "duplicate_id"),
        ("unreachable_node.json", "unreachable_from_start"),
        ("invalid_gateway_default_flow.json", "invalid_gateway_default_flow"),
        ("invalid_cross_scope_sequence_flow.json", "cross_scope_sequence_flow"),
    ],
)
def test_invalid_fixtures_report_specific_errors(
    fixture_name: str, expected_code: str
) -> None:
    assert expected_code in issue_codes(load_json(fixture_name))


def test_schema_validation_is_separate_and_forbids_unknown_fields() -> None:
    data = load_json("linear_process.json")
    data["unexpected"] = True

    with pytest.raises(ValidationError) as exc_info:
        ProcessModel.model_validate(data)

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
    assert exc_info.value.errors()[0]["loc"] == ("unexpected",)


def test_graph_error_exposes_json_location() -> None:
    model = ProcessModel.model_validate(load_json("invalid_dangling_sequence_flow.json"))

    with pytest.raises(ProcessGraphValidationError) as exc_info:
        validate_process_graph(model)

    issue = next(
        issue
        for issue in exc_info.value.issues
        if issue.code == "unknown_sequence_target"
    )
    assert issue.location == ("sequence_flows", 0, "target_ref")


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("unknown_lane", "unknown_lane"),
        ("start_incoming", "start_has_incoming"),
        ("end_outgoing", "end_has_outgoing"),
        ("same_pool_message", "same_participant_message_flow"),
        ("missing_condition", "missing_gateway_condition"),
    ],
)
def test_additional_structural_rules(mutation: str, expected_code: str) -> None:
    if mutation == "missing_condition":
        data = load_json("exclusive_gateway.json")
        data["nodes"][1].pop("default_flow_id")
        data["sequence_flows"][2]["condition"] = None
    else:
        data = load_json("linear_process.json")
        if mutation == "unknown_lane":
            data["nodes"][1]["lane_id"] = "missing_lane"
        elif mutation == "start_incoming":
            data["sequence_flows"].append(
                {"id": "back_to_start", "source_ref": "review", "target_ref": "start"}
            )
        elif mutation == "end_outgoing":
            data["sequence_flows"].append(
                {"id": "from_end", "source_ref": "end", "target_ref": "review"}
            )
        else:
            data["message_flows"] = [
                {"id": "internal_message", "source_ref": "start", "target_ref": "review"}
            ]

    assert expected_code in issue_codes(data)


def test_validation_never_repairs_or_changes_input() -> None:
    data = load_json("invalid_gateway_default_flow.json")
    original = deepcopy(data)
    model = ProcessModel.model_validate(data)

    with pytest.raises(ProcessGraphValidationError):
        validate_process_graph(model)

    assert data == original


@pytest.mark.parametrize(
    "fixture_name",
    [
        "two_participants_message.json",
        "nested_subprocess.json",
        "loop_corrections.json",
    ],
)
def test_hardened_invariants_preserve_valid_models(fixture_name: str) -> None:
    model = ProcessModel.model_validate(load_json(fixture_name))

    validate_process_graph(model)


def test_participant_requires_a_root_scope() -> None:
    data = load_json("linear_process.json")
    data["participants"].append({"id": "unmapped", "name": "Unmapped party"})

    model = ProcessModel.model_validate(data)
    with pytest.raises(ProcessGraphValidationError) as exc_info:
        validate_process_graph(model)

    issue = next(
        issue
        for issue in exc_info.value.issues
        if issue.code == "missing_participant_root_scope"
    )
    assert issue.location == ("participants", 1, "id")


def test_participant_rejects_multiple_root_scopes() -> None:
    data = load_json("linear_process.json")
    data["scopes"].append({"id": "second_root", "participant_id": "company"})

    assert "multiple_participant_root_scopes" in issue_codes(data)


def test_subprocess_requires_one_nested_scope() -> None:
    data = load_json("linear_process.json")
    data["nodes"][1]["type"] = "subprocess"

    assert "missing_subprocess_scope" in issue_codes(data)


def test_subprocess_rejects_duplicate_nested_scopes() -> None:
    data = load_json("nested_subprocess.json")
    data["scopes"].append(
        {
            "id": "duplicate_verification_scope",
            "participant_id": "company",
            "parent_scope_id": "main",
            "subprocess_node_id": "verification",
        }
    )

    assert "duplicate_subprocess_scope" in issue_codes(data)


def test_scope_hierarchy_rejects_parent_cycle() -> None:
    data = load_json("nested_subprocess.json")
    data["scopes"][0]["parent_scope_id"] = "main"
    data["scopes"][0]["subprocess_node_id"] = "verification"

    assert "scope_parent_cycle" in issue_codes(data)


def _add_nested_scope_for_message_subprocess(data: dict[str, Any]) -> None:
    data["nodes"][1]["type"] = "subprocess"
    data["scopes"].append(
        {
            "id": "send_order_scope",
            "participant_id": "customer",
            "parent_scope_id": "customer_process",
            "subprocess_node_id": "send_order",
        }
    )
    data["nodes"].extend(
        [
            {"id": "order_start", "type": "start_event", "name": "Order preparation started", "participant_id": "customer", "scope_id": "send_order_scope"},
            {"id": "prepare_order", "type": "task", "name": "Prepare order", "participant_id": "customer", "scope_id": "send_order_scope"},
            {"id": "order_end", "type": "end_event", "name": "Order preparation completed", "participant_id": "customer", "scope_id": "send_order_scope"},
        ]
    )
    data["sequence_flows"].extend(
        [
            {"id": "of1", "source_ref": "order_start", "target_ref": "prepare_order"},
            {"id": "of2", "source_ref": "prepare_order", "target_ref": "order_end"},
        ]
    )


@pytest.mark.parametrize("endpoint_kind", ["task", "subprocess", "participant"])
def test_eligible_message_flow_endpoints(endpoint_kind: str) -> None:
    data = load_json("two_participants_message.json")
    if endpoint_kind == "subprocess":
        _add_nested_scope_for_message_subprocess(data)
    elif endpoint_kind == "participant":
        data["message_flows"][0]["source_ref"] = "customer"
        data["message_flows"][0]["target_ref"] = "supplier"

    model = ProcessModel.model_validate(data)
    validate_process_graph(model)


@pytest.mark.parametrize(
    ("endpoint_field", "endpoint_id", "expected_code"),
    [
        ("source_ref", "customer_start", "ineligible_message_source"),
        ("source_ref", "customer_end", "ineligible_message_source"),
        ("target_ref", "supplier_start", "ineligible_message_target"),
        ("target_ref", "supplier_end", "ineligible_message_target"),
        ("target_ref", "receive_order", "ineligible_message_target"),
    ],
)
def test_message_flow_rejects_event_and_gateway_node_endpoints(
    endpoint_field: str, endpoint_id: str, expected_code: str
) -> None:
    data = load_json("two_participants_message.json")
    if endpoint_id == "receive_order":
        data["nodes"][4]["type"] = "parallel_gateway"
    data["message_flows"][0][endpoint_field] = endpoint_id

    assert expected_code in issue_codes(data)
