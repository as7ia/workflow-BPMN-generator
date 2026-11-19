"""Offline tests for supervised dataset construction and leakage guards."""

import json
from pathlib import Path

import pytest

from training.build_supervised_dataset import DEFAULT_SOURCE, build_dataset, build_examples
from training.train_qlora_kaggle import environment_report


def test_seed_examples_are_valid_diverse_synthetic_and_not_reserved() -> None:
    examples = build_examples()

    assert len(examples) == 6
    assert {item["metadata"]["family"] for item in examples} == {
        "linear",
        "exclusive_decision",
        "parallel",
        "loop",
        "collaboration",
        "subprocess",
    }
    assert all(item["metadata"]["provenance"]["synthetic"] for item in examples)
    assert all(len(item["messages"]) == 3 for item in examples)
    assert all(item["messages"][2]["content"].startswith("{") for item in examples)


def test_dataset_builder_writes_deterministic_train_validation_and_summary(tmp_path: Path) -> None:
    summary = build_dataset(DEFAULT_SOURCE, tmp_path)

    assert summary["reserved_evaluation_cases"] == 12
    assert summary["total_examples"] == 6
    assert summary["train_examples"] + summary["validation_examples"] == 6
    assert set(summary["unsplittable_families"]) == {
        "linear", "exclusive_decision", "parallel", "loop", "collaboration", "subprocess"
    }
    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "validation.jsonl").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_builder_rejects_duplicate_training_descriptions(tmp_path: Path) -> None:
    source = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
    duplicate = json.loads(json.dumps(source["cases"][0]))
    duplicate["id"] = "duplicate_description"
    duplicate["model"]["id"] = "different_model_id"
    source["cases"].append(duplicate)
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate description"):
        build_examples(path)


def test_preflight_reports_real_data_and_never_assumes_cuda(tmp_path: Path) -> None:
    build_dataset(DEFAULT_SOURCE, tmp_path)
    report = environment_report(tmp_path / "train.jsonl", tmp_path / "validation.jsonl")

    assert report["data"]["train_examples"] > 0
    assert report["data"]["validation_examples"] > 0
    assert isinstance(report["cuda"]["available"], bool)
    assert report["ready"] == (
        all(version is not None for version in report["packages"].values())
        and report["cuda"]["available"]
    )
