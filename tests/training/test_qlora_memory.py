"""Offline tests for the single-T4 QLoRA memory profile."""

import pytest

from training.train_qlora_kaggle import (
    memory_training_settings,
    precision_for_compute_capability,
    tokenize_supervised_record,
)


class CharacterTokenizer:
    """Small deterministic tokenizer with a prefix-preserving chat template."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        assert tokenize is False
        rendered = "".join(f"<{item['role']}>{item['content']}\n" for item in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(
        self, text: str, *, truncation: bool, add_special_tokens: bool
    ) -> dict[str, list[int]]:
        assert truncation is False
        assert add_special_tokens is False
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }


def _record(response: str = '{"schema_version":"1.0"}') -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Return ProcessModel JSON."},
            {"role": "user", "content": "Describe the process."},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"id": "memory-test"},
    }


def test_t4_compute_capability_forces_fp16_even_if_bf16_probe_might_succeed() -> None:
    assert precision_for_compute_capability(7, 5) == "fp16"
    assert precision_for_compute_capability(8, 0) == "bf16"


def test_only_assistant_tokens_contribute_to_loss() -> None:
    tokenizer = CharacterTokenizer()
    encoded = tokenize_supervised_record(tokenizer, _record(), max_length=8192)
    first_target = encoded["labels"].index(next(label for label in encoded["labels"] if label != -100))

    assert all(label == -100 for label in encoded["labels"][:first_target])
    assert encoded["labels"][first_target:] == encoded["input_ids"][first_target:]
    prompt_text = tokenizer.apply_chat_template(
        _record()["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    assert first_target == len(prompt_text)


def test_full_sequence_is_accepted_at_limit_and_never_truncated() -> None:
    tokenizer = CharacterTokenizer()
    baseline = tokenize_supervised_record(tokenizer, _record("x"), max_length=8192)
    response_length = 8192 - len(baseline["input_ids"]) + 1

    exact = tokenize_supervised_record(
        tokenizer, _record("x" * response_length), max_length=8192
    )
    assert len(exact["input_ids"]) == 8192
    with pytest.raises(ValueError, match="refusing to truncate the BPMN response"):
        tokenize_supervised_record(
            tokenizer, _record("x" * (response_length + 1)), max_length=8192
        )


def test_single_t4_memory_parameters_are_fixed_and_use_fused_loss() -> None:
    settings = memory_training_settings("fp16")

    assert settings["per_device_train_batch_size"] == 1
    assert settings["gradient_accumulation_steps"] == 8
    assert settings["fp16"] is True
    assert settings["bf16"] is False
    assert settings["gradient_checkpointing"] is True
    assert settings["use_liger_kernel"] is True
    assert settings["liger_kernel_config"] == {"fused_linear_cross_entropy": True}
