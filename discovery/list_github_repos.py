"""
List every repo in a GitHub org — the first building block of kb_graph's
discovery module (resolving a code-level reference, e.g. a Terraform module
source, against an org's actual repo list is step one of "find the repo this
points at").

Python port of a reference Node.js script (axios + manual pagination) aimed
at an internal GitHub Enterprise (GHE) instance, not github.com. Kept
stdlib-only (urllib.request) to match the rest of kb_graph — no new
dependency for something `urllib` already does fine.

GHE differs from public GitHub in two ways that matter here:
- API base path is `https://<host>/api/v3/...`, not `https://api.github.com/...`.
- It's common for an internal GHE host to sit behind a self-signed/internal
  CA cert that isn't in the standard trust store.

That second point is why the reference script disabled TLS verification
outright (`rejectUnauthorized: false`). This port does NOT do that by
default — turning off certificate verification is a real security
tradeoff, not a formality, and it should be an explicit opt-in
(`GITHUB_INSECURE_TLS=1`) rather than something baked in silently. The
verified-by-default/opt-out-by-env-var policy itself lives in
`common/net.py` so the Anthropic-gateway client shares the same rule instead
of a second copy of it.

Usage:
    export GITHUB_PAT=ghp_...
    export GITHUB_HOST=git.example.com    # omit for public github.com
    python discovery/list_github_repos.py my-org
    python discovery/list_github_repos.py my-org --json
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

# graph_model.py / common/ live one directory up (kb_graph/), not inside
# discovery/ — add it to sys.path the same way harness.py does, so this
# still works whether it's run as `python discovery/list_github_repos.py`
# or imported from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.net import build_ssl_context  # noqa: E402

PER_PAGE = 100


def api_base(host: str | None) -> str:
    if not host or host == "github.com":
        return "https://api.github.com"
    # GitHub Enterprise Server mounts the REST API under /api/v3.
    return f"https://{host}/api/v3"



def fetch_page(base_url: str, org: str, page: int, pat: str, ctx: ssl.SSLContext | None) -> list[dict]:
    url = f"{base_url}/orgs/{org}/repos?per_page={PER_PAGE}&page={page}&type=all"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {pat}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "kb_graph-discovery",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit("error: GitHub API returned 401 — check GITHUB_PAT is set and valid.")
        if e.code == 404:
            raise SystemExit(f"error: org '{org}' not found (404) — check the org name and host.")
        raise SystemExit(f"error: GitHub API returned HTTP {e.code}: {e.reason}")


def list_org_repos(org: str, host: str | None, pat: str) -> list[dict]:
    base_url = api_base(host)
    ctx = build_ssl_context("GITHUB_INSECURE_TLS")
    repos: list[dict] = []
    page = 1
    while True:
        print(f"fetching page {page}...", file=sys.stderr)
        batch = fetch_page(base_url, org, page, pat, ctx)
        repos.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return repos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("org", nargs="?", default=os.environ.get("GITHUB_ORG"))
    parser.add_argument("--json", action="store_true", help="print full repo objects as JSON")
    args = parser.parse_args()

    if not args.org:
        raise SystemExit("error: pass an org name, or set GITHUB_ORG.")

    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise SystemExit("error: set GITHUB_PAT to a valid personal access token.")

    host = os.environ.get("GITHUB_HOST")
    repos = list_org_repos(args.org, host, pat)

    if args.json:
        slim = [
            {
                "name": r["name"],
                "full_name": r["full_name"],
                "description": r.get("description"),
                "private": r["private"],
                "html_url": r["html_url"],
                "default_branch": r.get("default_branch"),
                "updated_at": r.get("updated_at"),
            }
            for r in repos
        ]
        print(json.dumps(slim, indent=2))
        return

    print(f"\n{len(repos)} repos in {args.org}:\n", file=sys.stderr)
    for r in repos:
        vis = "private" if r["private"] else "public"
        print(f"  {r['full_name']:<50} [{vis}]  {r.get('description') or ''}")


if __name__ == "__main__":
    main()
