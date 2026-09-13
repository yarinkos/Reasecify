"""
Shared TLS handling for kb_graph's stdlib HTTP clients.

Factored out of `discovery/list_github_repos.py`'s original `build_ssl_context`
so the same "verified by default, insecure only via an explicit env var"
policy is written once and reused by every outbound HTTPS call kb_graph
makes — the GitHub API (`discovery/list_github_repos.py`) and the Anthropic
gateway (`common/llm_client.py`) alike, both of which may sit behind an
internal host with a self-signed/internal CA cert.

Turning off certificate verification is a real security tradeoff, not a
formality — it must always be an explicit opt-in per env var, never a
default, and it always prints a warning when used.
"""

from __future__ import annotations

import os
import ssl
import sys


def build_ssl_context(insecure_env_var: str) -> ssl.SSLContext | None:
    """None => the caller's HTTPS client uses its normal, verified default
    context. Only returns an unverified context if the caller has set
    `insecure_env_var=1` — e.g. `GITHUB_INSECURE_TLS` for GitHub calls,
    a separate var for the Anthropic gateway, so opting out of verification
    for one host never silently opts out the other."""
    if os.environ.get(insecure_env_var) == "1":
        print(
            f"warning: {insecure_env_var}=1 — skipping TLS certificate "
            "verification for this host. Only use this for a known internal "
            "host with a self-signed/internal CA cert.",
            file=sys.stderr,
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None
