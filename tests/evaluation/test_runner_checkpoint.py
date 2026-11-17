"""Fast mock-only checks for interruption-safe evaluation reports."""

import json
import sys
from pathlib import Path

from app.domain import ProcessModel
from evaluation import run_qwen_evaluation as runner


def test_runner_checkpoints_and_resumes_without_repeating(monkeypatch, tmp_path, capsys):
    fixture = Path(__file__).parents[1] / "fixtures" / "linear_process.json"
    model = ProcessModel.model_validate(json.loads(fixture.read_text(encoding="utf-8")))
    calls = []

    def fake_extract(*args, **kwargs):
        calls.append(args[0])
        return model

    monkeypatch.setattr(runner, "extract_process", fake_extract)
    monkeypatch.setattr(runner, "score_candidate", lambda *a: {"overall": 0.5})
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    output = tmp_path / "sample.json"
    base = ["runner", "--confirm-live-ollama", "--max-cases", "2", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", base)
    assert runner.main() == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 2
    assert len(calls) == 2
    assert "[1/2] START" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", base + ["--resume"])
    assert runner.main() == 0
    assert len(calls) == 2


def test_runner_requires_explicit_consent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner"])
    import pytest

    with pytest.raises(SystemExit, match="confirm-live-ollama"):
        runner.main()
