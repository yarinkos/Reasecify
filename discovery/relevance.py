"""
Relevance scoring for the discovery module.

Pure functions, no I/O — deliberately kept separate from harness.py so the
scoring rule itself is easy to reason about (and test) in isolation from
GitHub API calls or Terraform parsing.

Same spirit as ingest.py's `link_documents` keyword-overlap rule: no
hardcoded "this is a generic module name" denylist. A module named
`auth_module` naturally shares zero words with a topic like "proxy farm" and
scores 0 — nothing topic-specific has to be taught to the scorer for that to
work. Longer/more shared words count for more, so a single generic overlap
word doesn't equal a real match.
"""

from __future__ import annotations

import re

# Superset of ingest.py's WORD_SPLIT_RE: a topic is free text a person types
# ("proxy farm stickiness"), not a slug, so this also splits on whitespace.
TOKEN_SPLIT_RE = re.compile(r"[\s/\-_.]+")

# Below this length a word is almost always structural noise ("tf", "the",
# "for") rather than topic signal - dropped rather than scored.
MIN_TOKEN_LEN = 3


def tokenize(text: str) -> set[str]:
    """Lowercase, split on whitespace/hyphen/underscore/dot/slash, drop
    anything shorter than MIN_TOKEN_LEN. Used for both topic text and
    module names/source strings so the two sides compare on equal footing."""
    return {
        w for w in TOKEN_SPLIT_RE.split(text.lower())
        if w and len(w) >= MIN_TOKEN_LEN
    }


def topic_relevance(topic_words: set[str], candidate_words: set[str]) -> float:
    """Sum of shared-word lengths — a stand-in for "how much of the topic
    does this candidate actually talk about." Rewards multiple/longer
    matches over one short coincidental overlap, same rationale as
    ingest.py's MIN_SOLO_KEYWORD_LEN check for `documents` edges. Returns
    0.0 for no overlap at all (the caller drops these, not this function —
    keeps this function a pure score, not a filter)."""
    shared = topic_words & candidate_words
    return float(sum(len(w) for w in shared))
