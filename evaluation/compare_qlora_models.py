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
from app.llm.ollama import OllamaSettings, _repair_payload, _system_prompt
from app.services import generate_bpmn
from evaluation.controlled_repair import apply_safe_deterministic_repairs
from evaluation.constrained_decoding import (
    ProcessModelConstraint,
    compile_process_model_constraint,
)
from evaluation.scoring import score_candidate
from training.train_qlora_kaggle import precision_for_compute_capability

CORPUS = ROOT / "evaluation" / "corpus.json"


def _generate_token_ids(
    model: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    pad_token_id: int,
    constraint: ProcessModelConstraint | None,
) -> Any:
    """Call generate with a fresh token-level constraint processor when enabled."""
    constrained_kwargs = constraint.generation_kwargs() if constraint is not None else {}
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        **constrained_kwargs,
    )


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


def business_preservation_check(
    model: ProcessModel, expected: dict[str, Any]
) -> dict[str, Any]:
    """Reject a valid repair that loses any reserved business requirement."""
    fidelity = score_candidate(model, expected)
    categories = fidelity["categories"]
    required_categories = (
        "participants_and_lanes",
        "gateway_types_and_branches",
        "parallel_split_join",
        "loop_preservation",
        "message_flow_direction",
        "subprocess_nesting",
    )
    failures = []
    if fidelity["missing_activities"]:
        failures.append({
            "code": "missing_business_activities",
            "values": fidelity["missing_activities"],
        })
    for category in required_categories:
        if categories[category] < 1.0:
            failures.append({
                "code": "business_requirement_not_preserved",
                "category": category,
                "score": categories[category],
            })
    return {"passed": not failures, "failures": failures, "fidelity": fidelity}


def evaluate_cases_with_repair(
    cases: list[dict[str, Any]],
    generate: Callable[[str], str],
    *,
    repair: Callable[[str, Any, str, list[dict[str, Any]]], str] | None,
    with_layout: bool,
) -> list[dict[str, Any]]:
    """Measure initial output and at most one fully revalidated repair attempt."""
    results: list[dict[str, Any]] = []
    for case in cases:
        content = generate(case["description"])
        before = assess_content(content, case["expected"], with_layout=with_layout)
        record: dict[str, Any] = {
            "id": case["id"],
            "category": case["category"],
            "before_repair": before,
            "after_repair": None,
            "deterministic_before_generation": None,
            "deterministic_after_generation": None,
            "repair": {
                "attempted": False,
                "validation_stage": None,
                "accepted": False,
                "reason": "not_needed" if before["graph_valid"] is True else "disabled",
                "business_preservation": None,
            },
        }
        error = before.get("error") or {}
        stage = error.get("stage")
        if repair is not None and before["graph_valid"] is not True:
            if stage not in {"schema", "graph"}:
                record["repair"]["reason"] = "unsupported_initial_failure_stage"
            else:
                try:
                    candidate = json.loads(content)
                except json.JSONDecodeError:
                    record["repair"]["reason"] = "initial_candidate_not_json"
                else:
                    normalized, deterministic = apply_safe_deterministic_repairs(
                        candidate
                    )
                    record["deterministic_before_generation"] = deterministic
                    normalized_content = json.dumps(normalized, separators=(",", ":"))
                    normalized_assessment = assess_content(
                        normalized_content,
                        case["expected"],
                        with_layout=with_layout,
                    )
                    if normalized_assessment["graph_valid"] is True:
                        normalized_model = ProcessModel.model_validate(normalized)
                        preservation = business_preservation_check(
                            normalized_model, case["expected"]
                        )
                        record["after_repair"] = normalized_assessment
                        record["repair"]["business_preservation"] = preservation
                        if preservation["passed"]:
                            record["repair"].update({
                                "accepted": True,
                                "reason": "deterministic_repair_validated_and_business_preserved",
                            })
                        else:
                            record["repair"]["reason"] = "business_preservation_failed"
                        record["effective_after_repair"] = (
                            normalized_assessment
                            if record["repair"]["accepted"]
                            else before
                        )
                        results.append(record)
                        continue
                    normalized_error = normalized_assessment.get("error") or {}
                    normalized_stage = normalized_error.get("stage")
                    if normalized_stage not in {"schema", "graph"}:
                        record["repair"]["reason"] = "deterministic_result_not_repairable"
                        record["effective_after_repair"] = before
                        results.append(record)
                        continue
                    issues = normalized_error.get("issues", [])
                    record["repair"].update({
                        "attempted": True,
                        "validation_stage": normalized_stage,
                        "reason": None,
                    })
                    try:
                        repaired_content = repair(
                            case["description"], normalized, normalized_stage, issues
                        )
                    except Exception as repair_error:
                        record["repair"].update({
                            "reason": "repair_generation_failed",
                            "error_type": type(repair_error).__name__,
                        })
                        record["effective_after_repair"] = before
                        results.append(record)
                        continue
                    try:
                        generated_candidate = json.loads(repaired_content)
                    except json.JSONDecodeError:
                        normalized_generated_content = repaired_content
                    else:
                        generated_candidate, generated_deterministic = (
                            apply_safe_deterministic_repairs(generated_candidate)
                        )
                        record["deterministic_after_generation"] = generated_deterministic
                        normalized_generated_content = json.dumps(
                            generated_candidate, separators=(",", ":")
                        )
                    after = assess_content(
                        normalized_generated_content,
                        case["expected"],
                        with_layout=with_layout,
                    )
                    record["after_repair"] = after
                    if after["graph_valid"] is True:
                        repaired_model = ProcessModel.model_validate(
                            json.loads(normalized_generated_content)
                        )
                        preservation = business_preservation_check(
                            repaired_model, case["expected"]
                        )
                        record["repair"]["business_preservation"] = preservation
                        if preservation["passed"]:
                            record["repair"].update({
                                "accepted": True,
                                "reason": "validated_and_business_preserved",
                            })
                        else:
                            record["repair"]["reason"] = "business_preservation_failed"
                    else:
                        record["repair"]["reason"] = "repaired_candidate_invalid"
        record["effective_after_repair"] = (
            record["after_repair"]
            if record["repair"]["accepted"]
            else before
        )
        results.append(record)
    return results


def summarize_repair_pipeline(results: list[dict[str, Any]]) -> dict[str, Any]:
    before = summarize([item["before_repair"] for item in results])
    after = summarize([item["effective_after_repair"] for item in results])
    initially_invalid = sum(
        item["before_repair"]["graph_valid"] is not True for item in results
    )
    accepted = sum(item["repair"]["accepted"] for item in results)
    return {
        "before_repair": before,
        "after_repair": after,
        "initially_invalid": initially_invalid,
        "repair_attempted": sum(item["repair"]["attempted"] for item in results),
        "generation_repair_attempted": sum(
            item["repair"]["attempted"] for item in results
        ),
        "repair_accepted": accepted,
        "repair_rejected": sum(
            item["repair"]["attempted"] and not item["repair"]["accepted"]
            for item in results
        ),
        "repair_rejected_total": initially_invalid - accepted,
        "deterministic_repair_accepted": sum(
            item["repair"]["reason"]
            == "deterministic_repair_validated_and_business_preserved"
            for item in results
        ),
        "deterministic_changes": sum(
            len((item.get(key) or {}).get("changes", []))
            for item in results
            for key in (
                "deterministic_before_generation",
                "deterministic_after_generation",
            )
        ),
    }


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
        "semantic_scored_cases": len(semantic),
        "semantic_coverage_rate": len(semantic) / count,
        "mean_semantic_fidelity": sum(semantic) / len(semantic) if semantic else None,
        "mean_semantic_fidelity_on_valid_graphs": (
            sum(semantic) / len(semantic) if semantic else None
        ),
    }


def _load_generator(
    model_name: str,
    adapter: Path | None,
    *,
    max_new_tokens: int,
    constrained_decoding: bool = False,
) -> tuple[
    Callable[[str], str],
    Callable[[list[dict[str, str]]], str],
    Any,
    Any,
    dict[str, Any],
]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    precision = precision_for_compute_capability(properties.major, properties.minor)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if precision == "bf16" else torch.float16
        ),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map={"": device_index},
        trust_remote_code=False,
    )
    model_vocab_size = model.config.vocab_size
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    constraint: ProcessModelConstraint | None = None
    constraint_report: dict[str, Any] = {"enabled": False}
    if constrained_decoding:
        constraint = compile_process_model_constraint(
            tokenizer, vocab_size=model_vocab_size
        )
        constraint_report = constraint.report

    def generate_messages(messages: list[dict[str, str]]) -> str:
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
            output = _generate_token_ids(
                model,
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                constraint=constraint,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(description: str) -> str:
        return generate_messages([
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": description},
        ])

    return generate, generate_messages, model, tokenizer, constraint_report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and QLoRA Qwen3-8B")
    parser.add_argument("--confirm-gpu-evaluation", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/qlora-comparison.json"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--with-layout", action="store_true")
    parser.add_argument(
        "--constrained-decoding",
        action="store_true",
        help="mask generation logits with the ProcessModel JSON Schema via XGrammar",
    )
    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help="allow exactly one controlled repair for schema or graph failures",
    )
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
        generate, generate_messages, model, tokenizer, constraint_report = _load_generator(
            args.model,
            adapter,
            max_new_tokens=args.max_new_tokens,
            constrained_decoding=args.constrained_decoding,
        )
        repair = None
        if args.repair_invalid:
            repair_settings = OllamaSettings(
                model=args.model,
                num_predict=args.max_new_tokens,
            )

            def repair(
                description: str,
                candidate: Any,
                stage: str,
                issues: list[dict[str, Any]],
            ) -> str:
                payload = _repair_payload(
                    description,
                    candidate,
                    issues,
                    repair_settings,
                    validation_stage=stage,
                )
                return generate_messages(payload["messages"])

        results = evaluate_cases_with_repair(
            cases,
            generate,
            repair=repair,
            with_layout=args.with_layout,
        )
        runs[label] = {
            "summary": summarize_repair_pipeline(results),
            "results": results,
        }
        del generate, generate_messages, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "reserved_corpus": str(CORPUS.resolve()),
        "case_ids": [case["id"] for case in cases],
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "constrained_decoding": constraint_report,
        },
        "with_layout": args.with_layout,
        "repair_invalid": args.repair_invalid,
        "maximum_generation_attempts_per_case": 2 if args.repair_invalid else 1,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in runs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
