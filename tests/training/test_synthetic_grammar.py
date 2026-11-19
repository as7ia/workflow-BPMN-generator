"""Quality and determinism tests for coupled synthetic process generation."""

import json
from pathlib import Path

import pytest

from app.domain import ProcessModel, validate_process_graph
from app.domain.process_model import NodeType
from training.build_supervised_dataset import _normalize, build_dataset
from training.generate_synthetic_grammar import (
    DOMAINS,
    generate_references,
    sequence_case,
    topology_fingerprint,
)


@pytest.fixture(scope="module")
def generated() -> tuple[dict, dict]:
    return generate_references()


def test_generation_is_valid_unique_and_reports_actual_acceptance(generated: tuple[dict, dict]) -> None:
    document, report = generated
    assert report["candidate_examples"] == 483
    assert report["accepted_examples"] == 482
    assert report["rejection_reasons"] == {"reserved_topology_equivalent": 1}
    assert report["unique_topologies"] == 482
    assert len({case["topology_fingerprint"] for case in document["cases"]}) == 482
    for case in document["cases"]:
        model = ProcessModel.model_validate(case["model"])
        validate_process_graph(model)


def test_every_business_element_is_present_in_its_coupled_description(generated: tuple[dict, dict]) -> None:
    document, _ = generated
    for case in document["cases"]:
        model = ProcessModel.model_validate(case["model"])
        text = _normalize(case["description"])
        required = [participant.name for participant in model.participants]
        required += [node.name for node in model.nodes if node.type in {NodeType.TASK, NodeType.SUBPROCESS}]
        required += [flow.condition for flow in model.sequence_flows if flow.condition]
        required += [flow.label for flow in model.message_flows if flow.label]
        assert all(_normalize(value) in text for value in required), case["id"]


def test_topology_fingerprint_ignores_business_vocabulary() -> None:
    _, first, _ = sequence_case(DOMAINS[0], 4)
    _, second, _ = sequence_case(DOMAINS[1], 4)
    assert topology_fingerprint(first) == topology_fingerprint(second)


def test_jsonl_split_holds_out_complete_structure_groups(
    generated: tuple[dict, dict], tmp_path: Path
) -> None:
    document, _ = generated
    source = tmp_path / "references.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    summary = build_dataset(source, tmp_path / "dataset")
    assert summary["total_examples"] == 482
    assert summary["structure_group_overlap"] == []
    assert summary["train_examples"] == 297
    assert summary["validation_examples"] == 185
    assert summary["unsplittable_families"] == {}
    expected_families = {
        "collaboration", "exclusive_decision", "loop", "parallel", "sequence", "subprocess"
    }
    assert set(summary["train_families"]) == expected_families
    assert set(summary["validation_families"]) == expected_families
