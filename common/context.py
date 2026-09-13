"""
Retrieval context builder for kb_graph's chat feature.

Scores every node in a loaded graph against the latest chat message by
shared-word overlap — the exact same mechanism discovery/relevance.py
already uses to score a Terraform module against a topic, and the same
"no embeddings/LLM in v1" honesty the README states for doc-to-infra
linking. Reused rather than reinvented so kb_graph has one scoring rule,
not three slightly-different ones.

Retrieval is per-turn, latest-message-only: each question re-scores the
whole graph from just that message's words, with no memory of what was
retrieved for earlier turns in the conversation. A follow-up like "tell me
more about that" won't automatically re-pull the prior turn's nodes — only
the LLM's own memory of what it already said. This is a known v1
limitation (see kb_graph/README.md), not an oversight.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery.relevance import tokenize, topic_relevance  # noqa: E402

# How many lines of local-file context to show around a matched node's
# source.line — enough to see the block the node came from without
# dumping the whole file into the prompt.
SNIPPET_CONTEXT_LINES = 6

# File types the raw-text fallback below searches — deliberately the same
# ones ingest.py itself parses (see ingest.py's rglob("*.tf")), so this
# fallback only ever looks at files the graph could plausibly describe, not
# every file in a clone.
RAW_SEARCH_EXTENSIONS = (".tf",)

# Cap on how many raw-text hits get surfaced, and how many lines of context
# each one carries — kept small since these already compete for prompt
# space with the top-K node matches above them.
RAW_SEARCH_MAX_RESULTS = 8
RAW_SEARCH_CONTEXT_LINES = 2

# A design_doc's full body lives in attrs["text"] with no source.line to
# snippet around, so it needs its own excerpting — see _best_excerpt().
EXCERPT_WIDTH = 500       # chars shown for the winning excerpt
EXCERPT_WINDOW = 220      # chars per scored slice while searching for it
EXCERPT_STEP = 110        # slide between slices (50% overlap)


def _node_text(node: dict) -> str:
    """Text a node is scored against: name/label plus anything free-text
    in attrs (e.g. a design_doc's full stripped body) — attribute values
    that aren't strings (line numbers, booleans, nested dicts) are skipped
    rather than stringified, since coercing them adds noise, not signal."""
    parts = [node.get("name", ""), node.get("label", "")]
    for v in node.get("attrs", {}).values():
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def _read_snippet(node: dict, source_roots: list[Path]) -> str | None:
    """For a node whose source is a real file path, read a few lines
    around source.line straight off disk — free precision, since by the
    time chat is reachable the repo is already cloned locally (no GitHub
    API round-trip needed). Returns None if there's no path, no line, or
    the file isn't found under any known source root (e.g. it was in a
    repo that's since been removed)."""
    src = node.get("source") or {}
    rel_path = src.get("path")
    if not rel_path:
        return None

    for root in source_roots:
        candidate = root / rel_path
        if candidate.is_file():
            try:
                lines = candidate.read_text(errors="replace").splitlines()
            except OSError:
                return None
            line_no = src.get("line")
            if not line_no:
                return None
            start = max(0, line_no - 1 - SNIPPET_CONTEXT_LINES)
            end = min(len(lines), line_no + SNIPPET_CONTEXT_LINES)
            snippet = "\n".join(lines[start:end])
            return f"{rel_path}:{line_no}\n{snippet}"
    return None


def _raw_text_matches(
    query_words: set[str],
    source_roots: list[Path],
    already_shown: set[tuple[str, int]],
    max_results: int = RAW_SEARCH_MAX_RESULTS,
) -> list[str]:
    """Fallback for facts that live in the repos but never made it into any
    graph node — e.g. a literal value on a plain `key = "..."` line inside a
    module/resource block, which parsers/terraform.py never captures since
    it only records a handful of named attributes (`source`/`version` for
    module blocks). Node-based scoring (`build_context`'s main loop) can
    only ever surface what got parsed into a node's name/label/attrs; it has
    no way to find a fact the parser dropped on the floor.

    This instead tokenizes every line of every source file directly —
    same tokenize()/topic_relevance() rule as node scoring, just applied to
    raw file content instead of node text — so a match here only requires
    the word to appear literally in a file, not that some parser already
    turned it into a named node. `already_shown` skips (path, line) pairs
    already surfaced as a node snippet above, so the same line isn't shown
    twice.
    """
    if not query_words:
        return []

    # First pass: tokenize every candidate line once, keeping only lines
    # that share at least one query word with it — same relevance gate
    # topic_relevance() applies, just checked directly since we need the
    # actual shared-word set per line for the rarity weighting below.
    candidates: list[tuple[Path, Path, int, list[str], set[str]]] = []
    for root in source_roots:
        if not root.is_dir():
            continue
        for ext in RAW_SEARCH_EXTENSIONS:
            for path in sorted(root.rglob(f"*{ext}")):
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                try:
                    lines = path.read_text(errors="replace").splitlines()
                except OSError:
                    continue
                for i, line in enumerate(lines, start=1):
                    if (str(rel), i) in already_shown:
                        continue
                    shared = query_words & tokenize(line)
                    if shared:
                        candidates.append((root, rel, i, lines, shared))

    if not candidates:
        return []

    # Rank by rarity, not length or count. topic_relevance() (used for node
    # scoring) sums shared-word *length*, which is right for a small curated
    # set of node names where a longer word reliably means "more specific."
    # A raw sweep over every line in a repo doesn't have that property: a
    # label variable like "application_platform" gets assigned on nearly
    # every block in this repo, so its 11-letter word would swamp genuinely
    # rare, diagnostic words like "image" (5 letters, only a handful of
    # lines) under length-based scoring. So weight each shared word by how
    # few of the *candidate* lines contain it instead (inverse-line-
    # frequency, i.e. idf).
    #
    # And rank by the single rarest shared word (max idf weight), not the
    # sum across all shared words — summing lets two so-so words (e.g. a
    # near-stopword like "the" plus a moderately common one like
    # "instance") outscore one genuinely rare, on-topic word ("image"),
    # since query_words itself isn't stopword-filtered. Max-then-sum keeps
    # a single strong hit on top while still using the sum to break ties
    # between equally-specific lines.
    line_freq: dict[str, int] = {}
    for *_unused, shared in candidates:
        for w in shared:
            line_freq[w] = line_freq.get(w, 0) + 1
    total = len(candidates)

    def idf(word: str) -> float:
        return math.log(total / line_freq[word]) + 1

    scored = [
        (max(idf(w) for w in shared), sum(idf(w) for w in shared), root, rel, line_no, lines)
        for root, rel, line_no, lines, shared in candidates
    ]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    results = []
    seen: set[tuple[Path, Path, int]] = set()
    for top_score, _sum_score, root, rel, line_no, lines in scored:
        key = (root, rel, line_no)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, line_no - 1 - RAW_SEARCH_CONTEXT_LINES)
        end = min(len(lines), line_no + RAW_SEARCH_CONTEXT_LINES)
        snippet = "\n".join(lines[start:end])
        results.append(f"{root.name}/{rel}:{line_no} (score: {top_score:.1f})\n  ```\n  {snippet}\n  ```")
        if len(results) >= max_results:
            break
    return results


def _best_excerpt(text: str, query_words: set[str], width: int = EXCERPT_WIDTH) -> str:
    """Pick the `width`-char slice of `text` most relevant to the query,
    instead of always the first `width` chars. A design_doc's body has no
    source.line to snippet around like a code node does, so it used to just
    show text[:500] — fine for a short doc, but blind to the actual answer
    in a longer one if it happens to sit past the intro/overview paragraph
    (exactly what happened with the stickiness design doc: the sentence
    stating "a single cloud NAT with a shared IP address" starts at
    character 692). Slides a scoring window across the text the same
    tokenize()/topic_relevance() way node text is scored, then centers the
    shown excerpt on whichever window matched best.
    """
    if len(text) <= width:
        return text.strip()

    best_start, best_score = 0, -1.0
    for start in range(0, max(1, len(text) - EXCERPT_WINDOW), EXCERPT_STEP):
        chunk = text[start:start + EXCERPT_WINDOW]
        score = topic_relevance(query_words, tokenize(chunk))
        if score > best_score:
            best_score, best_start = score, start

    if best_score <= 0:
        # No window matched the query at all — fall back to the intro,
        # same behavior as before this function existed.
        return text[:width].strip()

    center = best_start + EXCERPT_WINDOW // 2
    start = max(0, center - width // 2)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def build_context(index, user_message: str, top_k: int = 15) -> str:
    """Render a text block to prepend as grounding for one chat turn:
    graph-level counts, then one card per top-K matched node (id, type,
    label, source pointer, and a local-file snippet when available).

    `index` is a query.GraphIndex (or anything exposing the same
    `.meta`/`.nodes` shape) already loaded from the workspace's
    graph.json.
    """
    query_words = tokenize(user_message)
    source_roots = [Path(s) for s in index.meta.get("sources", [])]

    scored = []
    for node in index.nodes.values():
        score = topic_relevance(query_words, tokenize(_node_text(node)))
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]

    lines = [
        f"Graph summary: {len(index.nodes)} nodes, "
        f"{sum(1 for _ in index.meta.get('repos', []))} repo(s) "
        f"({', '.join(index.meta.get('repos', [])) or '-'}), "
        f"{index.meta.get('confluence_pages', 0)} design doc(s).",
        "",
    ]

    # (path, line) pairs already shown via a node snippet below, so the
    # raw-text fallback further down doesn't repeat the same line twice.
    already_shown: set[tuple[str, int]] = set()

    if not top:
        lines.append(
            "No graph nodes matched this question by keyword overlap — "
            "there may be no relevant content in the ingested repos/docs."
        )
    else:
        lines.append(f"Top {len(top)} matching node(s):")
        for score, node in top:
            src = node.get("source") or {}
            pointer = src.get("url") or src.get("path") or ""
            header = f"- [{node['type']}] {node['label']} (id: {node['id']}, score: {score:.0f})"
            if pointer:
                header += f" — {pointer}"
            lines.append(header)

            snippet = _read_snippet(node, source_roots)
            if snippet:
                lines.append(f"  ```\n  {snippet}\n  ```")
                if src.get("path") and src.get("line"):
                    already_shown.add((src["path"], src["line"]))

            text_attr = node.get("attrs", {}).get("text")
            if isinstance(text_attr, str) and text_attr and not snippet:
                # design_doc nodes carry their full body in attrs["text"] but
                # have no source.line to snippet around — show the
                # best-matching excerpt (see _best_excerpt) instead of the
                # whole page.
                excerpt = _best_excerpt(text_attr, query_words)
                lines.append(f"  excerpt: {excerpt}")

    # Raw-text fallback always runs, even when node matching found plenty —
    # a fact can live in a repo without any node ever having been created
    # for the line it's on (see _raw_text_matches' docstring).
    raw_matches = _raw_text_matches(query_words, source_roots, already_shown)
    if raw_matches:
        lines.append("")
        lines.append(
            "Raw text match(es) found by searching source files directly "
            "(literal lines the graph has no node for, e.g. a plain "
            "attribute assignment the parser doesn't capture):"
        )
        lines.extend(f"- {m}" for m in raw_matches)

    return "\n".join(lines)
