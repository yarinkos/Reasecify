"""
Discovery harness: given one already-local root repo and a topic, find the
other repos in its GitHub org that its Terraform modules point at, ranked by
relevance to the topic.

    1. list every repo in the org (discovery/list_github_repos.py)
    2. read the root repo's Terraform module_call blocks (parsers/terraform.py)
    3. score each module against the topic (discovery/relevance.py) — a
       generic/unrelated module naturally scores 0 and drops out
    4. resolve each remaining module's `source` string against the org's
       repo list (same significant-word-subset rule ingest.py's
       link_calls_module uses, just run against the org's repo list instead
       of a set of already-cloned directory names)
    5. cap at max_repos, best first

Two `mode`s share that pipeline: "topic" (default) runs step 3 for real and
drops zero-score modules before step 4; "all" skips topic scoring entirely
(every module is a candidate) and resolves everything against the org repo
list, topic or no topic — "what does this repo reference in this org,
period," with nothing filtered out before resolution.

No cloning, no graph/KB generation here — this only ever answers "which
repos are worth looking at," per the explicit phasing the user asked for.
Callable directly (no server dependency) so a future CLI or test can reuse
it without going through discovery/server.py.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# graph_model.py / parsers/ live one directory up (kb_graph/), not inside
# discovery/ — add it to sys.path the same way this file would be found if
# it were run as `python discovery/harness.py` directly, so the import works
# regardless of whether the caller's cwd is kb_graph/ or discovery/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_model import Graph  # noqa: E402
from parsers.terraform import parse_repo  # noqa: E402
from ingest import significant_words  # noqa: E402

from list_github_repos import list_org_repos  # noqa: E402
from relevance import tokenize, topic_relevance  # noqa: E402


class DiscoveryError(Exception):
    """User-facing failure (bad path, unresolvable org, GitHub API error) —
    caught once at the server layer and turned into a JSON error response,
    same role as query.py's QueryError."""


VALID_MODES = ("topic", "all")


# Matches ingest.py's GENERIC_REPO_WORDS filtering via significant_words();
# imported directly rather than redefined so the two "does this module
# source point at repo X" checks (local-corpus vs org-repo-list) never drift
# apart from each other.


GIT_CONFIG_ORIGIN_SECTION_RE = re.compile(r'\[remote "origin"\]([^\[]*)', re.DOTALL)
GIT_CONFIG_URL_RE = re.compile(r'^\s*url\s*=\s*(\S+)', re.MULTILINE)
SSH_REMOTE_RE = re.compile(r'^[\w.-]+@[\w.-]+:([^/]+)/[^/]+?(?:\.git)?$')
HTTPS_REMOTE_RE = re.compile(r'^https?://[^/]+/([^/]+)/[^/]+?(?:\.git)?/?$')


def parse_org_from_git_config(root_repo: Path) -> Optional[str]:
    """Best-effort org name from `root_repo/.git/config`'s `origin` remote
    URL — handles both the SSH (`git@host:org/repo.git`) and HTTPS
    (`https://host/org/repo.git`) forms. Returns None rather than guessing
    if there's no .git/config, no origin remote, or the URL doesn't match
    either known shape."""
    config_path = root_repo / ".git" / "config"
    if not config_path.is_file():
        return None
    text = config_path.read_text(errors="replace")
    section_m = GIT_CONFIG_ORIGIN_SECTION_RE.search(text)
    section = section_m.group(1) if section_m else text
    url_m = GIT_CONFIG_URL_RE.search(section)
    if not url_m:
        return None
    url = url_m.group(1)
    for pattern in (SSH_REMOTE_RE, HTTPS_REMOTE_RE):
        m = pattern.match(url)
        if m:
            return m.group(1)
    return None


def resolve_org(root_repo: Path, org_override: Optional[str]) -> str:
    if org_override:
        return org_override
    parsed = parse_org_from_git_config(root_repo)
    if parsed:
        return parsed
    raise DiscoveryError(
        f"couldn't determine a GitHub org for {root_repo} — no --org given and "
        f"{root_repo}/.git/config has no recognizable origin remote URL"
    )


def read_root_repo_modules(root_repo: Path) -> list[dict]:
    """Every module_call block in root_repo with a module_source attr, via
    the same parser ingest.py uses — just run against one repo instead of a
    whole --source folder."""
    if not (root_repo / ".git").is_dir():
        raise DiscoveryError(f"{root_repo} doesn't look like a git repo (no .git dir)")
    graph = Graph()
    parse_repo(graph, root_repo, root_repo.parent)
    return [
        {"name": n.name, "label": n.label, "module_source": n.attrs["module_source"], "source": n.source.to_dict()}
        for n in graph.nodes
        if n.type == "module_call" and n.attrs.get("module_source")
    ]


def score_modules(modules: list[dict], topic_words: set[str]) -> list[dict]:
    """Attach a relevance score + the topic words that actually matched to
    each module, best first. Does NOT drop zero-overlap modules — callers
    that only want relevant ones filter on relevance_score > 0 themselves.
    Keeping the full list here is what lets the debug "every module
    scanned" view show *why* something like a `foundations-firewall-module`
    source never showed up as a match (score 0, no shared words) instead of
    it just silently never appearing anywhere."""
    scored = []
    for m in modules:
        candidate_words = tokenize(m["name"]) | tokenize(m["module_source"])
        matched = sorted(topic_words & candidate_words)
        scored.append({**m, "relevance_score": topic_relevance(topic_words, candidate_words), "topic_matched_words": matched})
    scored.sort(key=lambda m: m["relevance_score"], reverse=True)
    return scored


def match_repo_for_source(source: str, org_repos: list[dict]) -> Optional[dict]:
    """Same rule as ingest.py's link_calls_module, run the other direction:
    there we asked "does this source contain one of our already-cloned
    repos' names"; here we ask the same question against the org's full
    repo list instead. A repo matches only if *every* one of its
    significant words appears in the source string — see
    ingest.link_calls_module's docstring for why a looser fixed-count
    threshold produced a real false positive."""
    source_words = significant_words(source)
    for repo in org_repos:
        repo_words = significant_words(repo["name"])
        if repo_words and repo_words <= source_words:
            return repo
    return None


def run_discovery(
    root_repo: Path,
    topic: Optional[str],
    max_repos: int,
    org: Optional[str] = None,
    mode: str = "topic",
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise DiscoveryError(f"unknown mode {mode!r} — expected one of {VALID_MODES}")

    root_repo = root_repo.resolve()
    if not root_repo.is_dir():
        raise DiscoveryError(f"{root_repo} is not a directory")

    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise DiscoveryError("GITHUB_PAT is not set")
    host = os.environ.get("GITHUB_HOST")

    resolved_org = resolve_org(root_repo, org)

    try:
        org_repos = list_org_repos(resolved_org, host, pat)
    except SystemExit as e:
        raise DiscoveryError(str(e)) from e

    modules = read_root_repo_modules(root_repo)

    if mode == "topic":
        if not topic:
            raise DiscoveryError("topic is required in 'topic' mode")
        topic_words = tokenize(topic)
        if not topic_words:
            raise DiscoveryError("topic has no usable words (all too short/empty)")
        all_scored_modules = score_modules(modules, topic_words)
        scored_modules = [m for m in all_scored_modules if m["relevance_score"] > 0]
    else:
        # mode == "all" — every module is a candidate; topic plays no role
        # at all, so score against an empty word set (relevance_score comes
        # out 0.0 for everything — meaningless here, not "nothing matched").
        all_scored_modules = score_modules(modules, set())
        scored_modules = all_scored_modules

    # Every relevant module gets resolved — nothing is skipped just because
    # max_repos was already "reached" by an earlier module, and nothing is
    # dropped just because two modules (typically dev/stg/prd variants of the
    # same block) resolve to the same repo. Same-repo modules are grouped
    # onto one card instead of the first one silently winning, so all
    # `modules_relevant` end up visible somewhere in the response — either
    # under a repo card's `matched_modules`, or in `unresolved_modules`.
    # max_repos caps the number of *distinct repos* returned, applied last.
    repo_groups: dict[str, dict] = {}
    unresolved_modules: list[dict] = []

    for m in scored_modules:
        repo = match_repo_for_source(m["module_source"], org_repos)
        module_ref = {
            "label": m["label"],
            "module_source": m["module_source"],
            "relevance_score": m["relevance_score"],
            "topic_matched_words": m["topic_matched_words"],
        }
        if repo is None:
            unresolved_modules.append(module_ref)
            continue
        entry = repo_groups.get(repo["full_name"])
        if entry is None:
            # scored_modules is sorted best-first, so the first module to
            # reach a given repo carries that repo's best score.
            repo_groups[repo["full_name"]] = {
                "repo": repo["full_name"],
                "description": repo.get("description"),
                "html_url": repo["html_url"],
                "private": repo["private"],
                "relevance_score": m["relevance_score"],
                "matched_modules": [module_ref],
            }
        else:
            entry["matched_modules"].append(module_ref)

    matched = sorted(repo_groups.values(), key=lambda e: e["relevance_score"], reverse=True)[:max_repos]

    return {
        "mode": mode,
        "org": resolved_org,
        "root_repo": str(root_repo),
        "topic": topic,
        "org_repo_count": len(org_repos),
        "modules_considered": len(modules),
        "modules_relevant": len(scored_modules),
        "matched": matched,
        "unresolved_modules": unresolved_modules,
        # Full audit trail — every module_call in the root repo, including
        # the ones that scored 0 and were never even attempted against the
        # org's repo list. This is what the UI's debug checkbox shows, so
        # "why didn't X show up" always has a visible answer.
        "all_modules": [
            {
                "label": m["label"],
                "module_source": m["module_source"],
                "relevance_score": m["relevance_score"],
                "topic_matched_words": m["topic_matched_words"],
            }
            for m in all_scored_modules
        ],
    }
