"""Kaggle-oriented QLoRA training entry point; never imported by production code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

REQUIRED_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets")


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
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
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


def train(args: argparse.Namespace) -> None:
    report = environment_report(args.train_file, args.validation_file)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit("Training preflight failed; no training was started")
    if args.preflight_only:
        return

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

    if args.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
        except ImportError as error:
            raise SystemExit("Unsloth backend requested but unsloth is not installed") from error
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
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
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization,
            device_map="auto",
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

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )

    def tokenize(record: dict[str, Any]) -> dict[str, Any]:
        full_text = _chat_text(
            tokenizer, record["messages"], add_generation_prompt=False
        )
        prompt_text = _chat_text(
            tokenizer, record["messages"][:-1], add_generation_prompt=True
        )
        encoded = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)
        prompt = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_length)
        prompt_length = min(len(prompt["input_ids"]), len(encoded["input_ids"]))
        encoded["labels"] = (
            [-100] * prompt_length + encoded["input_ids"][prompt_length:]
        )
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
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
