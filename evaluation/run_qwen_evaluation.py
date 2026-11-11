"""Explicit, bounded live Qwen evaluation runner. Not used by pytest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import OllamaExtractionError, OllamaSettings, extract_process
from app.services import generate_bpmn
from evaluation.scoring import score_candidate

CORPUS = ROOT / "evaluation" / "corpus.json"
RESULTS = ROOT / "evaluation" / "results"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded live Qwen ProcessModel evaluation")
    parser.add_argument("--confirm-live-ollama", action="store_true", help="required safety acknowledgement")
    parser.add_argument("--model", help="override OLLAMA_MODEL for this evaluation")
    parser.add_argument("--timeout", type=float, help="override Ollama timeout seconds")
    parser.add_argument("--num-predict", type=int, help="override bounded output tokens")
    parser.add_argument("--num-ctx", type=int, help="override context window")
    parser.add_argument("--max-cases", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--with-layout", action="store_true", help="also run semantic serialization and DI layout")
    parser.add_argument("--output", type=Path, help="result JSON path; defaults under evaluation/results")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.confirm_live_ollama:
        raise SystemExit("Refusing inference: pass --confirm-live-ollama explicitly")
    settings = OllamaSettings.from_env()
    overrides = {
        key: value
        for key, value in {
            "model": args.model,
            "timeout_seconds": args.timeout,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        }.items()
        if value is not None
    }
    settings = replace(settings, **overrides)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    cases = corpus["cases"][: args.max_cases]
    results: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        record: dict[str, Any] = {
            "id": case["id"],
            "category": case["category"],
            "schema_valid": None,
            "graph_valid": None,
            "di_valid": None,
            "semantic_fidelity": None,
            "error": None,
        }
        try:
            model = extract_process(case["description"], settings=settings)
            record["schema_valid"] = True
            record["graph_valid"] = True
            record["candidate"] = model.model_dump(mode="json")
            record["semantic_fidelity"] = score_candidate(model, case["expected"])
            if args.with_layout:
                generate_bpmn(model)
                record["di_valid"] = True
        except OllamaExtractionError as error:
            record["schema_valid"] = error.code != "schema_validation_failed"
            record["graph_valid"] = False if error.code == "graph_validation_failed" else None
            record["error"] = {"stage": "extraction", "code": error.code, "message": str(error)}
        except Exception as error:  # runner records unexpected layout/runtime failures per case
            record["di_valid"] = False if args.with_layout else None
            record["error"] = {"stage": "layout" if args.with_layout else "runtime", "type": type(error).__name__, "message": str(error)}
        record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        results.append(record)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS / f"qwen-evaluation-{timestamp}.json"
    output = output.resolve()
    results_root = RESULTS.resolve()
    if results_root not in output.parents:
        raise SystemExit("Output must be inside evaluation/results")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at_utc": timestamp,
        "corpus_version": corpus["corpus_version"],
        "settings": {**asdict(settings), "base_url": settings.base_url},
        "with_layout": args.with_layout,
        "attempts_per_case": 1,
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
