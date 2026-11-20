"""No-model tests for strict base/fine-tuned evaluation helpers."""

import json
from pathlib import Path

from evaluation.compare_qlora_models import (
    assess_content,
    evaluate_cases_with_repair,
    summarize,
    summarize_repair_pipeline,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _expected() -> dict[str, object]:
    return {
        "participants": ["Company"],
        "lanes": ["Operations"],
        "activities": ["Review request"],
        "gateway_counts": {"exclusive_gateway": 0, "parallel_gateway": 0},
        "parallel_blocks": [],
        "loops": [],
        "message_flows": [],
        "subprocesses": [],
    }


def test_assessment_keeps_schema_graph_and_semantics_distinct() -> None:
    valid = (FIXTURES / "linear_process.json").read_text(encoding="utf-8")
    invalid_graph = (FIXTURES / "invalid_dangling_sequence_flow.json").read_text(encoding="utf-8")

    valid_result = assess_content(valid, _expected(), with_layout=False)
    graph_result = assess_content(invalid_graph, _expected(), with_layout=False)
    malformed_result = assess_content("not json", _expected(), with_layout=False)

    assert valid_result["schema_valid"] is True
    assert valid_result["graph_valid"] is True
    assert valid_result["semantic_fidelity"]["overall"] == 1.0
    assert graph_result["schema_valid"] is True
    assert graph_result["graph_valid"] is False
    assert malformed_result["schema_valid"] is False
    assert malformed_result["graph_valid"] is None


def test_summary_uses_the_same_denominator_for_model_comparison() -> None:
    results = [
        {"schema_valid": True, "graph_valid": True, "di_valid": True, "semantic_fidelity": {"overall": 0.75}},
        {"schema_valid": False, "graph_valid": None, "di_valid": None, "semantic_fidelity": None},
    ]

    summary = summarize(results)

    assert summary == {
        "cases": 2,
        "schema_valid_rate": 0.5,
        "graph_valid_rate": 0.5,
        "di_valid_rate": 0.5,
        "semantic_scored_cases": 1,
        "semantic_coverage_rate": 0.5,
        "mean_semantic_fidelity": 0.75,
        "mean_semantic_fidelity_on_valid_graphs": 0.75,
    }


def _case(expected: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "id": "repair-case",
        "category": "linear",
        "description": "Company Operations reviews a request.",
        "expected": expected or _expected(),
    }


def test_graph_failure_gets_one_revalidated_business_preserving_repair() -> None:
    invalid = (FIXTURES / "invalid_dangling_sequence_flow.json").read_text(encoding="utf-8")
    valid = (FIXTURES / "linear_process.json").read_text(encoding="utf-8")
    calls = []

    def repair(description, candidate, stage, issues):
        calls.append((description, candidate, stage, issues))
        return valid

    results = evaluate_cases_with_repair(
        [_case()], lambda _: invalid, repair=repair, with_layout=False
    )
    result = results[0]

    assert len(calls) == 1
    assert calls[0][2] == "graph"
    assert result["before_repair"]["graph_valid"] is False
    assert result["after_repair"]["graph_valid"] is True
    assert result["repair"]["accepted"] is True
    assert result["effective_after_repair"]["graph_valid"] is True
    summary = summarize_repair_pipeline(results)
    assert summary["before_repair"]["graph_valid_rate"] == 0.0
    assert summary["after_repair"]["graph_valid_rate"] == 1.0


def test_schema_failure_is_distinct_and_can_be_repaired_once() -> None:
    invalid = json.loads((FIXTURES / "linear_process.json").read_text(encoding="utf-8"))
    invalid["unexpected_field"] = True
    valid = (FIXTURES / "linear_process.json").read_text(encoding="utf-8")
    stages = []

    results = evaluate_cases_with_repair(
        [_case()],
        lambda _: json.dumps(invalid),
        repair=lambda _description, _candidate, stage, _issues: stages.append(stage) or valid,
        with_layout=False,
    )

    assert stages == ["schema"]
    assert results[0]["before_repair"]["error"]["stage"] == "schema"
    assert results[0]["repair"]["accepted"] is True


def test_valid_repair_is_rejected_when_required_business_activity_disappears() -> None:
    invalid = (FIXTURES / "invalid_dangling_sequence_flow.json").read_text(encoding="utf-8")
    valid = (FIXTURES / "linear_process.json").read_text(encoding="utf-8")
    expected = _expected()
    expected["activities"] = ["Review request", "Preserve audit task"]

    result = evaluate_cases_with_repair(
        [_case(expected)], lambda _: invalid,
        repair=lambda *_: valid, with_layout=False,
    )[0]

    assert result["after_repair"]["graph_valid"] is True
    assert result["repair"]["accepted"] is False
    assert result["repair"]["reason"] == "business_preservation_failed"
    assert result["effective_after_repair"]["graph_valid"] is False
    assert result["repair"]["business_preservation"]["failures"][0]["code"] == "missing_business_activities"


def test_malformed_generated_repair_is_reported_and_rejected() -> None:
    invalid = (FIXTURES / "invalid_dangling_sequence_flow.json").read_text(encoding="utf-8")

    result = evaluate_cases_with_repair(
        [_case()], lambda _: invalid, repair=lambda *_: "{broken",
        with_layout=False,
    )[0]

    assert result["repair"]["attempted"] is True
    assert result["repair"]["accepted"] is False
    assert result["repair"]["reason"] == "repaired_candidate_invalid"
    assert result["after_repair"]["error"]["stage"] == "json"
