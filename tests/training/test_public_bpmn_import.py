"""Offline tests for the strict public BPMN subset importer."""

import json
from pathlib import Path

import pytest

from app.domain import ProcessModel, validate_process_graph
from training.public_bpmn_import import (
    ImportRejection,
    bpmn_to_process_model,
    import_camunda,
    write_import,
)


def test_camunda_import_reads_local_pair_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = tmp_path / "BPMN for Research" / "English" / "01-Test"
    exercise = case / "01-Exercise"
    solution = case / "03-Solution"
    exercise.mkdir(parents=True)
    solution.mkdir(parents=True)
    (exercise / "exercise.pdf").write_bytes(b"offline-test-placeholder")
    (solution / "solution.bpmn").write_text(_supported_bpmn(), encoding="utf-8")
    monkeypatch.setattr(
        "training.public_bpmn_import._extract_pdf_text",
        lambda _: "The Support participant must handle the request.",
    )

    cases, report = import_camunda(tmp_path, accept_research_only_license=True)

    assert len(cases) == 1
    assert report["candidate_pairs"] == 1
    assert report["accepted"] == 1
    validate_process_graph(ProcessModel.model_validate(cases[0]["model"]))

BPMN_HEADER = '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">'


def _supported_bpmn() -> str:
    return BPMN_HEADER + """
      <collaboration id="c"><participant id="p" name="Support" processRef="proc" /></collaboration>
      <process id="proc" name="Support process">
        <startEvent id="start" name="Request received" />
        <task id="task" name="Handle request" />
        <endEvent id="end" name="Request handled" />
        <sequenceFlow id="f1" sourceRef="start" targetRef="task" />
        <sequenceFlow id="f2" sourceRef="task" targetRef="end" />
      </process>
    </definitions>"""


def test_supported_bpmn_subset_converts_and_validates() -> None:
    model = bpmn_to_process_model(_supported_bpmn(), source_id="public-example")

    validate_process_graph(model)
    assert [participant.name for participant in model.participants] == ["Support"]
    assert any(node.name == "Handle request" for node in model.nodes)


def test_unsupported_bpmn_is_rejected_without_lossy_conversion() -> None:
    xml = BPMN_HEADER + """
      <collaboration><participant id="p" name="Support" processRef="proc" /></collaboration>
      <process id="proc" name="Support process">
        <startEvent id="start"/><inclusiveGateway id="inclusive"/><endEvent id="end"/>
      </process></definitions>"""

    with pytest.raises(ImportRejection, match="unsupported_bpmn_element:inclusiveGateway"):
        bpmn_to_process_model(xml, source_id="unsupported")


def test_camunda_import_requires_explicit_research_only_acceptance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="research-only"):
        import_camunda(tmp_path, accept_research_only_license=False)


def test_report_records_blocked_unlicensed_and_unavailable_sources(tmp_path: Path) -> None:
    report = {
        "source": "camunda/bpmn-for-research",
        "candidate_pairs": 0,
        "accepted": 0,
        "rejected": 0,
        "families": {},
        "complexity": {},
    }
    write_import(tmp_path, [], report)
    saved = json.loads((tmp_path / "public_import_report.json").read_text(encoding="utf-8"))

    sources = {item["source"]: item for item in saved["sources"]}
    assert sources["zenodo:7783492"]["license"] == "unspecified"
    assert sources["zenodo:15857589"]["accepted"] == 0
    assert sources["MaD_via_2025_study"]["accessibility"] == "not_publicly_downloadable"
