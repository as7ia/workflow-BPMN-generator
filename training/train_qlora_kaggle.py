"""Kaggle-oriented QLoRA training entry point; never imported by production code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

REQUIRED_PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets",
    "liger-kernel",
)


def precision_for_compute_capability(major: int, minor: int) -> str:
    """Select a native training dtype; pre-Ampere GPUs must use FP16."""
    if (major, minor) >= (8, 0):
        return "bf16"
    return "fp16"


def memory_training_settings(precision: str) -> dict[str, Any]:
    """Centralize the single-GPU memory invariants used by TrainingArguments."""
    return {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "use_liger_kernel": True,
        "liger_kernel_config": {"fused_linear_cross_entropy": True},
    }


def environment_report(train_file: Path, validation_file: Path) -> dict[str, Any]:
    """Check actual packages, data, and CUDA before claiming training is possible."""
    versions: dict[str, str | None] = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    train_count = _count_valid_records(train_file)
    validation_count = _count_valid_records(validation_file)
    cuda: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "compute_capability": [
                        torch.cuda.get_device_properties(index).major,
                        torch.cuda.get_device_properties(index).minor,
                    ],
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                    "selected_precision": precision_for_compute_capability(
                        torch.cuda.get_device_properties(index).major,
                        torch.cuda.get_device_properties(index).minor,
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return {
        "packages": versions,
        "data": {"train_examples": train_count, "validation_examples": validation_count},
        "cuda": cuda,
        "ready": (
            all(version is not None for version in versions.values())
            and train_count > 0
            and validation_count > 0
            and cuda["available"]
        ),
    }


def _count_valid_records(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("messages"), list) or not record.get("metadata", {}).get("provenance"):
            raise ValueError(f"{path}:{line_number}: missing messages or provenance")
        count += 1
    return count


def _chat_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def tokenize_supervised_record(
    tokenizer: Any, record: dict[str, Any], *, max_length: int
) -> dict[str, list[int]]:
    """Tokenize without truncation and mask every non-assistant target token."""
    full_text = _chat_text(tokenizer, record["messages"], add_generation_prompt=False)
    prompt_text = _chat_text(
        tokenizer, record["messages"][:-1], add_generation_prompt=True
    )
    encoded = tokenizer(full_text, truncation=False, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
    input_ids = list(encoded["input_ids"])
    prompt_ids = list(prompt["input_ids"])
    record_id = record.get("metadata", {}).get("id", "unknown")
    if len(input_ids) > max_length:
        raise ValueError(
            f"{record_id}: tokenized sequence has {len(input_ids)} tokens, exceeding "
            f"max_seq_length={max_length}; refusing to truncate the BPMN response"
        )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{record_id}: chat-template prompt tokens are not an exact prefix of the "
            "full conversation; assistant-only loss masking would be unsafe"
        )
    assistant_length = len(input_ids) - len(prompt_ids)
    if assistant_length <= 0:
        raise ValueError(f"{record_id}: assistant response contains no trainable tokens")
    result = dict(encoded)
    result["input_ids"] = input_ids
    result["labels"] = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return result


def train(args: argparse.Namespace) -> None:
    report = environment_report(args.train_file, args.validation_file)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit("Training preflight failed; no training was started")
    if args.preflight_only:
        return
    if args.batch_size != 1 or args.gradient_accumulation != 8:
        raise SystemExit(
            "This single-T4 profile requires batch_size=1 and gradient_accumulation=8"
        )

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )

    def tokenize(record: dict[str, Any]) -> dict[str, Any]:
        return tokenize_supervised_record(
            tokenizer, record, max_length=args.max_seq_length
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    length_report = {
        split: {
            "examples": len(tokenized[split]),
            "maximum_tokens": max(len(item) for item in tokenized[split]["input_ids"]),
        }
        for split in ("train", "validation")
    }
    print(json.dumps({"tokenized_data": length_report}, indent=2))
    if args.validate_data_only:
        return

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    precision = precision_for_compute_capability(properties.major, properties.minor)
    compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    if args.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
        except ImportError as error:
            raise SystemExit("Unsloth backend requested but unsloth is not installed") from error
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=compute_dtype,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization,
            device_map={"": device_index},
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        **memory_training_settings(precision),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            padding=True,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_dir = args.output / "adapter-final"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA Qwen3-8B training for Kaggle GPU")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-file", type=Path, default=Path("training/artifacts/train.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("training/artifacts/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training/outputs/qwen3-8b-processmodel"))
    parser.add_argument("--backend", choices=("transformers", "unsloth"), default="transformers")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    train(_arguments())
