"""
LoRA fine-tuning, adapted from the llama-finetune-smoke-test project's
train_lora.py into a re-entrant function with progress reporting, instead of
a standalone script with hardcoded paths.

Every hyperparameter here is unchanged from the smoke test — this is a
straight port of a design that was already validated to change model
behavior on a small "design rationale" dataset, not a fresh design.

IMPORTANT (ported from the smoke test's own docstring): the incoming
dataset is expected to already bake in repeat-exposure (see
finetune/dataset_gen.py). Do NOT raise num_train_epochs above 1 — that
would multiply exposure on top of the dataset's own repeats.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from finetune.errors import FinetuneError  # noqa: E402

MAX_LENGTH = 512
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LEARNING_RATE = 1.5e-4
NUM_TRAIN_EPOCHS = 1  # DO NOT change — see module docstring
PER_DEVICE_TRAIN_BATCH_SIZE = 2


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_examples(path: Path) -> list[list[dict]]:
    import json
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line)["messages"])
    return examples


def _build_features(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Tokenize a single chat example, masking the prompt portion of the
    labels so loss is only computed on the assistant's answer — unchanged
    from the smoke test's train_lora.py."""
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels}


def _make_collate_fn(pad_token_id: int) -> Callable:
    import torch

    def collate(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def run_finetune(
    dataset_path: Path,
    model_id: str,
    out_dir: Path,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Fine-tune `model_id` on `dataset_path` (train_dataset.jsonl,
    "messages"-shaped lines) via LoRA, saving the adapter to `out_dir`.
    Calls `on_progress({...})` at each phase transition and training-loop
    log step so a caller (workflow/server.py's background job wrapper) can
    surface live status to a polling browser. Heavy imports (torch,
    transformers, peft) are deferred into this function so importing
    finetune/train.py itself stays cheap for callers that only need the
    other finetune/* modules (dataset_gen, model_cache) without the full
    training stack installed yet.
    """
    if not dataset_path.is_file():
        raise FinetuneError(f"dataset not found: {dataset_path}")

    def report(**kwargs):
        if on_progress is not None:
            on_progress(kwargs)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise FinetuneError(
            f"fine-tuning dependencies aren't installed ({e}) — "
            f"run: pip install -r kb_graph/requirements.txt"
        ) from e

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                report(
                    phase="training",
                    step=state.global_step,
                    total_steps=state.max_steps,
                    loss=logs["loss"],
                )

    device = _get_device()
    report(phase="loading", message=f"loading base model {model_id} on {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.to(device)
        model.config.use_cache = False
    except Exception as e:  # noqa: BLE001 — surface as a clean FinetuneError, never a stack trace
        raise FinetuneError(f"failed to load base model {model_id!r}: {e}") from e

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    report(phase="tokenizing", message=f"loading + tokenizing {dataset_path}...")
    raw_examples = _load_examples(dataset_path)
    if not raw_examples:
        raise FinetuneError(f"{dataset_path} has no training examples")
    features = [_build_features(m, tokenizer, MAX_LENGTH) for m in raw_examples]

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(features)

        def __getitem__(self, idx):
            return features[idx]

    dataset = _Dataset()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=(device != "cpu"),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_make_collate_fn(tokenizer.pad_token_id),
        callbacks=[_ProgressCallback()],
    )

    report(phase="training", message=f"starting training on {len(dataset)} examples...", step=0, total_steps=None)
    train_result = trainer.train()
    final_loss = getattr(train_result, "training_loss", None)

    report(phase="saving", message=f"saving LoRA adapter to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    report(phase="done", message="fine-tuning complete.")
    return {
        "adapter_dir": str(out_dir),
        "train_examples": len(dataset),
        "final_loss": final_loss,
    }
