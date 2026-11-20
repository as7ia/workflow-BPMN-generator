"""XGrammar-backed constrained decoding for ProcessModel evaluation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from app.domain import ProcessModel


@dataclass(frozen=True)
class ProcessModelConstraint:
    """A compiled constraint that creates fresh, stateful processors per request."""

    compiled_grammar: Any
    xgrammar: Any
    report: dict[str, Any]

    def generation_kwargs(self) -> dict[str, Any]:
        """Return kwargs that make Transformers mask logits during generation."""
        processor = self.xgrammar.contrib.hf.LogitsProcessor(self.compiled_grammar)
        return {"logits_processor": [processor]}


def _adapt_schema_for_xgrammar(schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Handle known XGrammar limitations without silently weakening the schema."""
    adapted = copy.deepcopy(schema)
    adaptations: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            if "pattern" in value and "minLength" in value:
                minimum = value.pop("minLength")
                adaptations.append({
                    "path": list(path + ("minLength",)),
                    "action": "omitted_during_decoding",
                    "reason": "xgrammar_pattern_and_min_length_conflict",
                    "original_value": minimum,
                    "post_validation": "ProcessModel.model_validate",
                })
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (index,))

    visit(adapted, ())
    return adapted, adaptations


def process_model_decoding_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the decoding schema and an explicit compatibility report."""
    source = ProcessModel.model_json_schema()
    adapted, adaptations = _adapt_schema_for_xgrammar(source)
    report = {
        "enabled": True,
        "engine": "xgrammar",
        "schema_source": "ProcessModel.model_json_schema()",
        "strict_mode": True,
        "property_order": "ProcessModel schema declaration order",
        "schema_adaptations": adaptations,
        "post_validation_only": [
            "identifier uniqueness across collections",
            "cross-reference existence and type correctness",
            "scope hierarchy and subprocess parent relationships",
            "start-to-end reachability within each scope",
            "gateway split/join and default-flow graph rules",
            "business-semantic preservation",
        ],
    }
    return adapted, report


def compile_process_model_constraint(
    tokenizer: Any,
    vocab_size: int,
    *,
    xgrammar_module: Any | None = None,
) -> ProcessModelConstraint:
    """Compile ProcessModel JSON Schema for token-level constrained decoding."""
    if xgrammar_module is None:
        try:
            import xgrammar as xgrammar_module
        except ImportError as error:
            raise RuntimeError(
                "--constrained-decoding requires xgrammar; install the Kaggle "
                "evaluation requirements before loading the model"
            ) from error

    schema, report = process_model_decoding_schema()
    tokenizer_info = xgrammar_module.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size
    )
    compiler = xgrammar_module.GrammarCompiler(tokenizer_info)
    try:
        compiled = compiler.compile_json_schema(json.dumps(schema))
    except Exception as error:
        raise RuntimeError(
            "XGrammar could not compile the adapted ProcessModel schema; "
            "constrained evaluation was not started"
        ) from error
    return ProcessModelConstraint(compiled, xgrammar_module, report)
