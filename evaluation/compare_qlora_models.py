"""Compare base and QLoRA-adapted Qwen on the reserved evaluation corpus."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from app.domain import ProcessGraphValidationError, ProcessModel, validate_process_graph
from app.llm.ollama import _system_prompt
from app.services import generate_bpmn
from evaluation.scoring import score_candidate

CORPUS = ROOT / "evaluation" / "corpus.json"


def assess_content(
    content: str,
    expected: dict[str, Any],
    *,
    with_layout: bool,
) -> dict[str, Any]:
    """Apply the same strict schema, graph, DI, and semantic checks to one output."""
    record: dict[str, Any] = {
        "schema_valid": False,
        "graph_valid": None,
        "di_valid": None,
        "semantic_fidelity": None,
        "error": None,
    }
    try:
        candidate = json.loads(content)
    except json.JSONDecodeError:
        record["error"] = {"stage": "json", "code": "malformed_output"}
        return record
    try:
        model = ProcessModel.model_validate(candidate)
    except ValidationError as error:
        record["error"] = {
            "stage": "schema",
            "code": "schema_validation_failed",
            "issues": [
                {"location": list(item["loc"]), "type": item["type"]}
                for item in error.errors()
            ],
        }
        return record
    record["schema_valid"] = True
    try:
        validate_process_graph(model)
    except ProcessGraphValidationError as error:
        record["graph_valid"] = False
        record["error"] = {
            "stage": "graph",
            "code": "graph_validation_failed",
            "issues": [
                {"code": issue.code, "location": list(issue.location)}
                for issue in error.issues
            ],
        }
        return record
    record["graph_valid"] = True
    record["semantic_fidelity"] = score_candidate(model, expected)
    if with_layout:
        try:
            generate_bpmn(model)
        except Exception as error:
            record["di_valid"] = False
            record["error"] = {"stage": "layout", "type": type(error).__name__}
        else:
            record["di_valid"] = True
    return record


def evaluate_cases(
    cases: list[dict[str, Any]],
    generate: Callable[[str], str],
    *,
    with_layout: bool,
) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        result = {"id": case["id"], "category": case["category"]}
        result.update(assess_content(generate(case["description"]), case["expected"], with_layout=with_layout))
        results.append(result)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    semantic = [
        item["semantic_fidelity"]["overall"]
        for item in results
        if item["semantic_fidelity"] is not None
    ]
    return {
        "cases": count,
        "schema_valid_rate": sum(item["schema_valid"] for item in results) / count,
        "graph_valid_rate": sum(item["graph_valid"] is True for item in results) / count,
        "di_valid_rate": (
            sum(item["di_valid"] is True for item in results) / count
            if any(item["di_valid"] is not None for item in results)
            else None
        ),
        "mean_semantic_fidelity": sum(semantic) / len(semantic) if semantic else None,
    }


def _load_generator(
    model_name: str,
    adapter: Path | None,
    *,
    max_new_tokens: int,
) -> tuple[Callable[[str], str], Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=False,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(description: str) -> str:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": description},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return generate, model, tokenizer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and QLoRA Qwen3-8B")
    parser.add_argument("--confirm-gpu-evaluation", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/qlora-comparison.json"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--with-layout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.confirm_gpu_evaluation:
        raise SystemExit("Pass --confirm-gpu-evaluation explicitly")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available; evaluation was not started")
    if not args.adapter.is_dir():
        raise SystemExit("Adapter directory does not exist")
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]
    runs: dict[str, Any] = {}
    for label, adapter in (("base", None), ("fine_tuned", args.adapter)):
        generate, model, tokenizer = _load_generator(
            args.model, adapter, max_new_tokens=args.max_new_tokens
        )
        results = evaluate_cases(cases, generate, with_layout=args.with_layout)
        runs[label] = {"summary": summarize(results), "results": results}
        del generate, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "reserved_corpus": str(CORPUS.resolve()),
        "case_ids": [case["id"] for case in cases],
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "with_layout": args.with_layout,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in runs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
