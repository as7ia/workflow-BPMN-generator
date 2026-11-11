"""No-inference checks for the Qwen semantic evaluation corpus."""

import json
from pathlib import Path

from app.domain import ProcessModel, validate_process_graph
from evaluation.scoring import score_candidate

ROOT = Path(__file__).parents[2]
CORPUS = ROOT / "evaluation" / "corpus.json"


def _cases() -> list[dict[str, object]]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]


def test_corpus_has_twelve_unique_bounded_english_cases() -> None:
    cases = _cases()

    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
    assert all(40 <= len(case["description"]) <= 12_000 for case in cases)
    assert {
        "linear",
        "exclusive_decision",
        "parallel",
        "loop",
        "collaboration",
        "subprocess",
        "combined",
        "combined_collaboration",
        "combined_subprocess_loop",
    } <= {case["category"] for case in cases}


def test_every_reference_model_is_valid_and_matches_manual_expectations() -> None:
    for case in _cases():
        reference_path = ROOT / case["reference"]
        data = json.loads(reference_path.read_text(encoding="utf-8"))
        model = ProcessModel.model_validate(data)

        validate_process_graph(model)
        score = score_candidate(model, case["expected"])

        assert score["overall"] == 1.0, case["id"]
        assert score["missing_activities"] == [], case["id"]
        assert score["invented_activities"] == [], case["id"]


def test_all_rubric_dimensions_are_declared_per_case() -> None:
    required = {
        "participants",
        "lanes",
        "activities",
        "gateway_counts",
        "parallel_blocks",
        "loops",
        "message_flows",
        "subprocesses",
    }
    for case in _cases():
        assert required <= set(case["expected"]), case["id"]
