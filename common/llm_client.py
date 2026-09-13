"""
Minimal stdlib Anthropic Messages API client for kb_graph's chat feature.

kb_graph is deliberately independent of the rest of GitDigger (see its
README) and has no `anthropic` SDK dependency. Rather than adding one just
for this one call site, this hand-rolls the same REST request the official
SDK sends under the hood — `POST {base_url}/v1/messages` with an `x-api-key`
header and a JSON body — which is exactly what the main app's `tools/llm.py`
does via `anthropic.Anthropic(base_url=...)`. Since the Requesty gateway is
already built to be a drop-in `base_url` for that official client, it
answers this raw request identically. Mirrors
`discovery/list_github_repos.py`'s own choice to hand-roll a thin GitHub
client instead of adding a GitHub SDK dependency.

Env vars (same names `config.py` already defines, so an existing `.env`
works unchanged — no import of `config.py` itself):
    ANTHROPIC_AUTH_TOKEN   required, the gateway API key
    ANTHROPIC_BASE_URL     default: the public Anthropic API — point this at
                           an internal gateway (e.g. a Requesty deployment)
                           via `.env` instead of hardcoding one here
    ANTHROPIC_MODEL        default: a public Claude model id — point this at
                           a gateway-specific policy alias via `.env` instead
                           of hardcoding one here
    ANTHROPIC_INSECURE_TLS opt-in only, see common/net.py — never a default
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from common.net import build_ssl_context

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-4-5"
ANTHROPIC_VERSION = "2023-06-01"


class ChatError(Exception):
    """User-facing failure (missing token, gateway error, malformed
    response) — same role as discovery's DiscoveryError / ingest's
    IngestError, caught once at the server layer and turned into a JSON
    error response instead of a stack trace."""


def ask(system_prompt: str, messages: list[dict], max_tokens: int = 2048) -> str:
    """Send a (possibly multi-turn) conversation to Claude and return the
    latest assistant reply as plain text.

    Unlike tools/llm.py's ask() (single-turn only — one system prompt, one
    user prompt), this takes the full `messages` list
    (`[{"role": "user"|"assistant", "content": "..."}]`) so a real chat can
    carry conversation history across turns — the caller (kb_graph/workflow)
    is expected to resend the growing history each turn; this function is
    stateless and makes no attempt to persist it.
    """
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not token:
        raise ChatError("ANTHROPIC_AUTH_TOKEN is not set")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=body,
        method="POST",
        headers={
            "x-api-key": token,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    ctx = build_ssl_context("ANTHROPIC_INSECURE_TLS")

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise ChatError(f"Anthropic gateway returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ChatError(f"couldn't reach Anthropic gateway at {base_url}: {e.reason}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ChatError(f"Anthropic gateway returned a non-JSON response: {e}") from e

    # `content[0]` isn't reliably the text block — with extended thinking on,
    # the gateway puts a `thinking` block first (and possibly
    # `redacted_thinking`/`tool_use` blocks too), so scan for every `text`
    # block instead of assuming position 0. A reply can also legitimately
    # span more than one text block; join them rather than taking just the
    # first.
    try:
        content = data["content"]
        text_parts = [
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    except (KeyError, TypeError) as e:
        raise ChatError(f"unexpected response shape from Anthropic gateway: {data!r}") from e
    if not text_parts:
        raise ChatError(f"no text content in Anthropic gateway response: {data!r}")
    return "".join(text_parts)
