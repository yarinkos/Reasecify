"""
Shared error type for kb_graph/finetune/* — same role as
discovery.harness.DiscoveryError / ingest.IngestError / common.clone.CloneError
/ common.llm_client.ChatError: a user-facing failure message, caught once at
the workflow server layer and turned into a clean JSON error response instead
of a stack trace reaching the browser.
"""

from __future__ import annotations


class FinetuneError(Exception):
    """Raised for bad input or a failure anywhere in the generate-data /
    check-cache / train / chat-with-fine-tuned-model pipeline."""
