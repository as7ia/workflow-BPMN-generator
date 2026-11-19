"""Build validated, leakage-checked Qwen chat examples from trusted references."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import ProcessModel, validate_process_graph
from app.llm.ollama import _system_prompt

DEFAULT_SOURCE = ROOT / "training" / "reference_cases.json"
RESERVED_CORPUS = ROOT / "evaluation" / "corpus.json"
DEFAULT_OUTPUT = ROOT / "training" / "artifacts"


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _semantic_fingerprint(model: ProcessModel) -> str:
    nodes = {node.id: node for node in model.nodes}
    payload = {
        "participants": sorted(_normalize(item.name) for item in model.participants),
        "lanes": sorted(_normalize(item.name) for item in model.lanes),
        "scopes": sorted(
            (
                scope.parent_scope_id is None,
                _normalize(next(p.name for p in model.participants if p.id == scope.participant_id)),
            )
            for scope in model.scopes
        ),
        "nodes": sorted((node.type.value, _normalize(node.name)) for node in model.nodes),
        "sequence_flows": sorted(
            (
                nodes[flow.source_ref].type.value,
                _normalize(nodes[flow.source_ref].name),
                nodes[flow.target_ref].type.value,
                _normalize(nodes[flow.target_ref].name),
                _normalize(flow.condition or ""),
            )
            for flow in model.sequence_flows
        ),
        "message_flows": sorted(
            (
                _normalize(nodes[flow.source_ref].name) if flow.source_ref in nodes else flow.source_ref,
                _normalize(nodes[flow.target_ref].name) if flow.target_ref in nodes else flow.target_ref,
                _normalize(flow.label or ""),
            )
            for flow in model.message_flows
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_reserved() -> tuple[list[str], set[str], set[str]]:
    corpus = json.loads(RESERVED_CORPUS.read_text(encoding="utf-8"))
    descriptions: list[str] = []
    fingerprints: set[str] = set()
    references: set[str] = set()
    for case in corpus["cases"]:
        descriptions.append(_normalize(case["description"]))
        reference = (ROOT / case["reference"]).resolve()
        references.add(str(reference))
        model = ProcessModel.model_validate(json.loads(reference.read_text(encoding="utf-8")))
        fingerprints.add(_semantic_fingerprint(model))
    return descriptions, fingerprints, references


def _assert_not_reserved(
    case: dict[str, Any],
    model: ProcessModel,
    reserved_descriptions: list[str],
    reserved_fingerprints: set[str],
) -> None:
    normalized = _normalize(case["description"])
    closest = max(
        SequenceMatcher(None, normalized, reserved).ratio()
        for reserved in reserved_descriptions
    )
    if closest >= 0.72:
        raise ValueError(f"{case['id']}: description resembles reserved evaluation text")
    if _semantic_fingerprint(model) in reserved_fingerprints:
        raise ValueError(f"{case['id']}: model duplicates a reserved evaluation process")


def build_examples(source: Path = DEFAULT_SOURCE) -> list[dict[str, Any]]:
    """Validate references and return deduplicated conversational examples."""
    document = json.loads(source.read_text(encoding="utf-8"))
    reserved_descriptions, reserved_fingerprints, _ = _load_reserved()
    seen_ids: set[str] = set()
    seen_descriptions: set[str] = set()
    seen_models: set[str] = set()
    examples: list[dict[str, Any]] = []
    for case in document["cases"]:
        provenance = case.get("provenance", {})
        if provenance.get("synthetic") is not True:
            raise ValueError(f"{case.get('id')}: synthetic provenance must be explicit")
        model = ProcessModel.model_validate(case["model"])
        validate_process_graph(model)
        _assert_not_reserved(case, model, reserved_descriptions, reserved_fingerprints)
        normalized_description = _normalize(case["description"])
        fingerprint = _semantic_fingerprint(model)
        if case["id"] in seen_ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        if normalized_description in seen_descriptions:
            raise ValueError(f"duplicate description: {case['id']}")
        if fingerprint in seen_models:
            raise ValueError(f"duplicate semantic model: {case['id']}")
        seen_ids.add(case["id"])
        seen_descriptions.add(normalized_description)
        seen_models.add(fingerprint)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": case["description"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            model.model_dump(mode="json"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "metadata": {
                    "id": case["id"],
                    "family": case["family"],
                    "structure_family": case.get("structure_family", case["family"]),
                    "split_group": case.get("split_group", case["family"]),
                    "schema_version": model.schema_version,
                    "model_fingerprint": fingerprint,
                    "provenance": provenance,
                },
            }
        )
    return examples


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _group_hash(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)


def _complexity_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        model = ProcessModel.model_validate(json.loads(record["messages"][2]["content"]))
        node_count = len(model.nodes)
        counts["simple" if node_count <= 8 else "medium" if node_count <= 16 else "complex"] += 1
    return dict(sorted(counts.items()))


def build_dataset(source: Path, output: Path) -> dict[str, Any]:
    examples = build_examples(source)
    groups = sorted({item["metadata"]["split_group"] for item in examples})
    validation_groups = {
        group
        for group in groups
        if _group_hash(group) % 5 == 0
    }
    groups_by_family: dict[str, set[str]] = {}
    for item in examples:
        family = item["metadata"]["family"]
        group = item["metadata"]["split_group"]
        groups_by_family.setdefault(family, set()).add(group)
    unsplittable_families: dict[str, list[str]] = {}
    for family, family_groups in sorted(groups_by_family.items()):
        if len(family_groups) < 2:
            unsplittable_families[family] = sorted(family_groups)
            continue
        selected = validation_groups & family_groups
        ordered = sorted(family_groups, key=lambda group: (_group_hash(group), group))
        if not selected:
            validation_groups.add(ordered[0])
        elif selected == family_groups:
            validation_groups.remove(ordered[-1])
    training = [item for item in examples if item["metadata"]["split_group"] not in validation_groups]
    validation = [item for item in examples if item["metadata"]["split_group"] in validation_groups]
    if not validation and len(training) > 1:
        validation.append(training.pop())
    _write_jsonl(output / "train.jsonl", training)
    _write_jsonl(output / "validation.jsonl", validation)
    summary = {
        "source": str(source.resolve()),
        "reserved_evaluation_cases": 12,
        "total_examples": len(examples),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "families": dict(sorted(Counter(item["metadata"]["family"] for item in examples).items())),
        "train_families": dict(sorted(Counter(item["metadata"]["family"] for item in training).items())),
        "validation_families": dict(sorted(Counter(item["metadata"]["family"] for item in validation).items())),
        "complexity": _complexity_counts(examples),
        "train_complexity": _complexity_counts(training),
        "validation_complexity": _complexity_counts(validation),
        "train_structure_groups": sorted({item["metadata"]["split_group"] for item in training}),
        "validation_structure_groups": sorted(validation_groups),
        "structure_group_overlap": sorted(
            {item["metadata"]["split_group"] for item in training}
            & {item["metadata"]["split_group"] for item in validation}
        ),
        "unsplittable_families": unsplittable_families,
        "all_references_validated": True,
        "all_examples_synthetic": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-checked QLoRA JSONL")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    print(json.dumps(build_dataset(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
