"""
Durable record of past fine-tuning runs, so Step 7's "chat with the
fine-tuned model" panel can offer a picker across server restarts instead
of only ever knowing about whatever job the *current* browser session just
finished.

This is a small JSON manifest file living at `<runs_dir>/manifest.json`
(`runs_dir` is workspace-scoped: `<workspace>/output/finetune/runs`, passed
in from the browser same as every other workspace-relative path in
workflow/server.py) — one record per job, appended when a fine-tune job
starts and updated in place when it finishes or fails. `_finetune_jobs` in
workflow/server.py stays exactly as it was (in-memory only) for the noisy
per-step training log a *running* job's poll loop wants; this module is
only for the small, durable summary a picker needs once a job is done.

Deliberately stdlib-only (json/pathlib/threading), like model_cache.py —
listing past runs shouldn't require the ML dependencies to be installed.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"

# One process-wide lock for all manifest reads/writes. Fine-tune runs are
# each minutes long and manifest updates are tiny, so serializing across
# every workspace's runs_dir costs nothing in practice and avoids the
# bookkeeping of a per-path lock table.
_lock = threading.Lock()


def _manifest_path(runs_dir: Path) -> Path:
    return runs_dir / MANIFEST_NAME


def _read_manifest_locked(runs_dir: Path) -> list[dict]:
    """Caller must hold `_lock`. Missing or corrupt manifest reads as
    empty — a from-scratch workspace or a hand-edited/truncated file
    shouldn't break the picker, just start it empty."""
    path = _manifest_path(runs_dir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _write_manifest_locked(runs_dir: Path, records: list[dict]) -> None:
    """Caller must hold `_lock`. Writes via a temp file + os.replace so a
    reader never sees a half-written manifest, even if two processes
    somehow pointed at the same runs_dir."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(runs_dir)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(records, indent=2))
    os.replace(tmp_path, path)


def start_run(runs_dir: Path, job_id: str, model_id: str, dataset_path: str) -> Path:
    """Records a new "running" entry for `job_id` and returns the
    per-run adapter directory it should be trained into
    (`<runs_dir>/<job_id>`) — one directory per run, rather than every
    fine-tune overwriting a single fixed path, so past adapters survive
    later runs and can actually be listed."""
    adapter_dir = runs_dir / job_id
    record = {
        "job_id": job_id,
        "model_id": model_id,
        "dataset_path": dataset_path,
        "adapter_dir": str(adapter_dir),
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_examples": None,
        "final_loss": None,
        "error": None,
    }
    with _lock:
        records = _read_manifest_locked(runs_dir)
        records.append(record)
        _write_manifest_locked(runs_dir, records)
    return adapter_dir


def finish_run(runs_dir: Path, job_id: str, result: dict) -> None:
    """Marks `job_id` done with `result` (the dict run_finetune() returned:
    adapter_dir/train_examples/final_loss)."""
    with _lock:
        records = _read_manifest_locked(runs_dir)
        for rec in records:
            if rec.get("job_id") == job_id:
                rec["status"] = "done"
                rec["train_examples"] = result.get("train_examples")
                rec["final_loss"] = result.get("final_loss")
                break
        _write_manifest_locked(runs_dir, records)


def fail_run(runs_dir: Path, job_id: str, error: str) -> None:
    """Marks `job_id` errored, so the picker can show it (grayed out)
    rather than silently dropping it."""
    with _lock:
        records = _read_manifest_locked(runs_dir)
        for rec in records:
            if rec.get("job_id") == job_id:
                rec["status"] = "error"
                rec["error"] = error
                break
        _write_manifest_locked(runs_dir, records)


def list_runs(runs_dir: Path) -> list[dict]:
    """Newest-first list of every run recorded under `runs_dir`, across
    however many server restarts happened in between."""
    with _lock:
        records = _read_manifest_locked(runs_dir)
    return sorted(records, key=lambda r: r.get("created_at") or "", reverse=True)
