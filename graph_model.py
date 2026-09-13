"""
Node/Edge data model for kb_graph.

This defines the one thing this whole prototype is really about getting right up
front: a deliberate, typed schema for the graph, instead of letting each parser
invent its own shape. Every node is source-anchored (it points back at a file+line
or a Confluence page id, never a copy of the original text) so the graph stays a
map to the sources rather than a stale duplicate of them.

Everything here serializes to plain dicts/lists so `graph.json` is just
`{"nodes": [...], "edges": [...], "meta": {...}}` — no custom decoder needed by
anything that consumes it later (a future query/lint step, or the HTML viewer).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Node types this v1 pipeline produces. Kept as plain strings (not an enum) so
# adding a new source type later doesn't require touching this module.
NODE_TYPES = {"repo", "file", "resource", "module_call", "variable", "output", "design_doc"}

# Edge types. See README for what each means; kept small and deliberate rather
# than letting parsers invent ad-hoc relationship names.
EDGE_TYPES = {"contains", "calls_module", "references", "documents"}


@dataclass
class Source:
    """Where a node's information actually lives — never copy the source text
    itself into the node, just point at it."""

    kind: str  # "file" | "confluence" | "external" (no location in this corpus)
    path: Optional[str] = None  # for kind == "file": path relative to --source root
    line: Optional[int] = None  # for kind == "file": 1-indexed line of the block header
    page_id: Optional[str] = None  # for kind == "confluence"
    url: Optional[str] = None  # for kind == "confluence"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Node:
    id: str
    type: str
    name: str
    label: str
    source: Source
    # Small bag of type-specific extras (e.g. repo name for a file node, resource
    # type for a resource node). Deliberately loose — this is a prototype schema,
    # not a locked-down one.
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "label": self.label,
            "source": self.source.to_dict(),
            "attrs": self.attrs,
        }


@dataclass
class Edge:
    source: str  # node id
    target: str  # node id
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "attrs": self.attrs,
        }


class Graph:
    """A simple in-memory accumulator. Nodes are deduped by id (last write wins,
    with a warning-free merge of attrs) so parsers can reference a node before or
    after it's been fully declared without worrying about ordering."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._edge_seen: set[tuple[str, str, str]] = set()

    def add_node(self, node: Node) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        else:
            # Merge attrs rather than clobber, in case two parsers touch the same
            # node id (e.g. a repo node declared once but referenced by many files).
            existing.attrs.update(node.attrs)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def add_edge(self, edge: Edge) -> None:
        key = (edge.source, edge.target, edge.type)
        if key in self._edge_seen:
            return
        self._edge_seen.add(key)
        self._edges.append(edge)

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self._nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        return counts

    def to_dict(self, meta: dict[str, Any]) -> dict:
        return {
            "meta": meta,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def node_id(type_: str, *parts: str) -> str:
    """Stable, readable id scheme: "type:part1/part2/...". Using real names
    (rather than a hash) keeps graph.json diffable and debuggable by hand."""
    slug = "/".join(p.strip("/") for p in parts if p)
    return f"{type_}:{slug}"
