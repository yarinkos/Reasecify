"""
Local Hugging Face cache check for the fine-tune step's UI — "is this model
already on disk, or is Start about to kick off a large first-time download?"

Deliberately stdlib-only (pathlib), no huggingface_hub import: this only
needs a yes/no + the path to show in the UI, not a full cache inventory, and
keeping it dependency-free means the check works even before `pip install -r
requirements.txt` has been run for the rest of the fine-tuning feature.
"""

from __future__ import annotations

import os
from pathlib import Path


def _hub_cache_dir() -> Path:
    """Same precedence huggingface_hub itself resolves the cache dir with:
    HF_HUB_CACHE, then HF_HOME/hub, then the default ~/.cache/huggingface/hub.
    """
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_cached(model_id: str) -> dict:
    """Returns {"cached": bool, "cache_dir": str} for `model_id`
    (e.g. "meta-llama/Llama-3.1-8B-Instruct"). `cached` is only True if the
    model's folder exists *and* has at least one non-empty snapshot — a
    folder left behind by an interrupted download (no files under
    snapshots/<hash>/) doesn't count, so the UI doesn't tell the user a
    multi-GB download is already done when it isn't.
    """
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = _hub_cache_dir() / folder_name
    snapshots_dir = model_dir / "snapshots"

    cached = False
    if snapshots_dir.is_dir():
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir() and any(snapshot.iterdir()):
                cached = True
                break

    return {"cached": cached, "cache_dir": str(model_dir)}
