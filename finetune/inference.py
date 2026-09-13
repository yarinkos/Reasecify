"""
Fine-tuned-model chat inference, adapted from the llama-finetune-smoke-test
project's query_finetuned_model.py.

An 8B-parameter model can't be loaded per chat request — this module keeps a
process-wide cache (`_loaded` below) so the merged model+tokenizer are loaded
at most once per (model_id, adapter_dir) pair, then reused for every
subsequent chat turn. This is the one piece of real server-side state
kb_graph/workflow introduces beyond the job-status tracker in
finetune/train.py's caller — a deliberate, scoped exception to the rest of
kb_graph's stateless-server convention (see workflow/server.py's module
docstring).

No retrieval/RAG context is injected here (unlike common/context.py's
chat-with-Claude path) — the whole point of this chat panel is to test
whether the fine-tuned *weights* recall the trained facts, not to re-do
retrieval a second time.
"""

from __future__ import annotations

import threading
from typing import Optional

from finetune.errors import FinetuneError

MAX_NEW_TOKENS = 400

_lock = threading.Lock()
_loaded: dict[tuple[str, str], tuple] = {}  # (model_id, adapter_dir) -> (model, tokenizer)


def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_or_load(model_id: str, adapter_dir: str) -> tuple:
    """Returns (model, tokenizer) for this (model_id, adapter_dir) pair,
    loading + merging the LoRA adapter into the base weights on first call
    and reusing the cached result on every subsequent call. Blocking is
    intentional and confined to the first chat request after a fine-tune
    run — the caller (workflow/server.py's /api/finetune/chat handler) is
    expected to show a "loading model" status for the duration of that one
    request rather than polling a background job for this step.
    """
    key = (model_id, adapter_dir)
    with _lock:
        if key in _loaded:
            return _loaded[key]

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise FinetuneError(
                f"fine-tuning dependencies aren't installed ({e}) — "
                f"run: pip install -r kb_graph/requirements.txt"
            ) from e

        device = _get_device()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            base_model.to(device)

            peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
            # merge_and_unload() works around a PeftModel.generate()
            # incompatibility with this transformers version — same
            # workaround the smoke test used.
            model = peft_model.merge_and_unload()
            model.to(device)
            model.eval()
        except Exception as e:  # noqa: BLE001 — never a stack trace to the browser
            raise FinetuneError(f"failed to load fine-tuned model ({model_id}, {adapter_dir}): {e}") from e

        _loaded[key] = (model, tokenizer)
        return model, tokenizer


def generate_reply(
    model,
    tokenizer,
    message: str,
    history: Optional[list[dict]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy-decodes a reply to `message` given the fine-tuned model —
    do_sample=False, matching the smoke test's own eval methodology, so
    results are reproducible run to run rather than sampled."""
    import torch

    messages = [*(history or []), {"role": "user", "content": message}]
    device = next(model.parameters()).device
    # return_dict=True explicitly, rather than relying on return_tensors="pt"
    # alone to hand back a bare tensor — newer transformers (the smoke test
    # this was ported from predates this) returns a BatchEncoding either
    # way, so asking for the dict form up front (and reading input_ids/
    # attention_mask out of it) works across versions instead of assuming
    # one particular return shape.
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
