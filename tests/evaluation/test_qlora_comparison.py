"""No-model tests for strict base/fine-tuned evaluation helpers."""

import json
from pathlib import Path

from evaluation.compare_qlora_models import assess_content, summarize

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
        "mean_semantic_fidelity": 0.75,
    }
