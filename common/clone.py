"""
git-clone helper for kb_graph's workflow module.

Deliberately not `tools/github_api.py:clone_repo` from the main GitDigger
app: that function's token-injection only rewrites `https://github.com/...`
URLs, so it silently clones *unauthenticated* against a GitHub Enterprise
host like `git.example.com` (private repos would just 404/permission-fail).
It also reads `GITHUB_TOKEN` — GitDigger's own default credential, not
kb_graph's. Per explicit instruction, kb_graph's clone step must keep using
kb_graph's own `GITHUB_PAT`/`GITHUB_HOST` env vars (the same ones
`discovery/list_github_repos.py` already requires) — never GitDigger's
`GITHUB_TOKEN`/`config.py` default — so this is a small, separate
implementation rather than a shared call into that function.

Shells out to the `git` CLI via `subprocess` (stdlib, no GitPython
dependency, matching the rest of kb_graph).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class CloneError(Exception):
    """User-facing failure (git not found, clone failed) — same role as
    DiscoveryError/IngestError/ChatError."""


def _authenticated_url(html_url: str, pat: str, host: str | None) -> str:
    """Embed the PAT in the HTTPS clone URL so `git clone` authenticates
    non-interactively, for whichever host GITHUB_HOST resolves to (or
    public github.com when unset) — same host GITHUB_HOST already
    resolves to for the REST API in list_github_repos.py."""
    target_host = host or "github.com"
    if f"://{target_host}/" not in html_url and not html_url.startswith(f"https://{target_host}"):
        # html_url came straight from the GitHub API response for this same
        # host, so this should never trip — but if it does, guessing at a
        # rewrite would risk cloning the wrong thing silently.
        raise CloneError(
            f"repo URL {html_url!r} doesn't match expected host {target_host!r}"
        )
    # https://<host>/... -> https://x-access-token:<pat>@<host>/...
    return html_url.replace(f"https://{target_host}", f"https://x-access-token:{pat}@{target_host}", 1)


def clone_repo(full_name: str, html_url: str, dest_dir: Path, pat: str, host: str | None) -> dict:
    """Clone one repo into `dest_dir / <repo-name>`.

    Idempotent: if the destination already has a `.git` directory, this is
    a no-op that reports `status: "already_present"` rather than re-cloning
    or erroring — matches `tools/github_api.py:clone_repo`'s own
    idempotency, just re-implemented here against kb_graph's own auth.
    """
    repo_name = full_name.rsplit("/", 1)[-1]
    target = dest_dir / repo_name

    if (target / ".git").is_dir():
        return {"repo": full_name, "path": str(target), "status": "already_present"}

    dest_dir.mkdir(parents=True, exist_ok=True)
    auth_url = _authenticated_url(html_url, pat, host)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", auth_url, str(target)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as e:
        raise CloneError("git executable not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise CloneError(f"git clone of {full_name} timed out") from e

    if result.returncode != 0:
        # stderr from `git clone` often contains the auth_url with the PAT
        # embedded (e.g. in a redirected-URL message) — scrub it so a PAT
        # never ends up in a log or an API error response.
        stderr = result.stderr.replace(pat, "***")
        raise CloneError(f"git clone of {full_name} failed: {stderr.strip()}")

    return {"repo": full_name, "path": str(target), "status": "cloned"}
