#!/usr/bin/env python3
"""
kb_graph query CLI.

The first real "operation" on the graph, per Karpathy's ingest/query/lint
model — `ingest.py` only builds `graph.json`; this reads it back and answers
actual questions ("what does web_service_001 depend on", "what does the
stickiness doc cover", "how are these two things connected") instead of
requiring you to pan around the canvas viewer.

Reads `graph.json` only — no changes to graph_model.py/parsers/ingest.py, and
no dependency on the exact objects those modules produce, just the plain
dict shape ingest.py already writes: {"meta", "nodes": [...], "edges": [...]}.

Every subcommand accepts a loose node reference (an id, an exact name, or a
substring of one) rather than requiring the full "type:path/to/thing" id —
see `resolve_node`. An ambiguous or unresolvable reference is a hard error,
never a silent guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

EDGE_DIRECTIONS = ("out", "in", "both")


class QueryError(Exception):
    """Raised for user-facing failures (bad ref, ambiguous match, bad graph
    file) — caught once in main() so every subcommand can just raise."""


class GraphIndex:
    """In-memory indices over a loaded graph.json for fast lookup/traversal.
    Built once per run; nothing here is persisted."""

    def __init__(self, data: dict):
        self.meta = data.get("meta", {})
        self.nodes: dict[str, dict] = {n["id"]: n for n in data.get("nodes", [])}
        self.edges: list[dict] = data.get("edges", [])

        self.out_edges: dict[str, list[dict]] = {}
        self.in_edges: dict[str, list[dict]] = {}
        for e in self.edges:
            self.out_edges.setdefault(e["source"], []).append(e)
            self.in_edges.setdefault(e["target"], []).append(e)

    def resolve_node(self, ref: str) -> dict:
        """Exact id -> exact name/label (case-insensitive) -> substring of
        name/label (case-insensitive). Raises QueryError with the candidate
        list if the substring stage is ambiguous, or if nothing matches."""
        if ref in self.nodes:
            return self.nodes[ref]

        ref_lower = ref.lower()
        exact = [n for n in self.nodes.values() if n["name"].lower() == ref_lower or n["label"].lower() == ref_lower]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise QueryError(self._ambiguous_message(ref, exact))

        substring = [n for n in self.nodes.values() if ref_lower in n["name"].lower() or ref_lower in n["label"].lower()]
        if len(substring) == 1:
            return substring[0]
        if len(substring) > 1:
            raise QueryError(self._ambiguous_message(ref, substring))

        raise QueryError(f'no node matches "{ref}"')

    @staticmethod
    def _ambiguous_message(ref: str, candidates: list[dict]) -> str:
        lines = [f'"{ref}" matches {len(candidates)} nodes — be more specific:']
        for n in sorted(candidates, key=lambda n: n["id"]):
            lines.append(f'  {n["id"]}  ({n["type"]})  {n["label"]}')
        return "\n".join(lines)

    def edges_for(self, node_id: str, direction: str, edge_type: Optional[str]) -> list[tuple[dict, str]]:
        """Returns (edge, direction_label) pairs — direction_label is "out" or
        "in" from node_id's perspective, kept even when direction="both" so
        callers can render an arrow the right way."""
        results: list[tuple[dict, str]] = []
        if direction in ("out", "both"):
            for e in self.out_edges.get(node_id, []):
                if edge_type is None or e["type"] == edge_type:
                    results.append((e, "out"))
        if direction in ("in", "both"):
            for e in self.in_edges.get(node_id, []):
                if edge_type is None or e["type"] == edge_type:
                    results.append((e, "in"))
        return results


def load_graph(path: Path) -> GraphIndex:
    if not path.is_file():
        raise QueryError(f"graph file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as ex:
        raise QueryError(f"could not parse {path} as JSON: {ex}")
    return GraphIndex(data)


def format_source(node: dict) -> str:
    src = node.get("source", {})
    if src.get("url"):
        return src["url"]
    if src.get("path"):
        line = f":{src['line']}" if src.get("line") else ""
        return f"{src['path']}{line}"
    return ""


def format_node_line(node: dict) -> str:
    src = format_source(node)
    return f'{node["id"]}  ({node["type"]})  {node["label"]}' + (f"  — {src}" if src else "")


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_find(idx: GraphIndex, args) -> Any:
    needle = args.text.lower()
    matches = [n for n in idx.nodes.values() if needle in n["name"].lower() or needle in n["label"].lower()]
    matches.sort(key=lambda n: n["id"])
    if args.json:
        return matches
    if not matches:
        print(f'no nodes match "{args.text}"')
        return None
    for n in matches:
        print(format_node_line(n))
    return None


def cmd_show(idx: GraphIndex, args) -> Any:
    node = idx.resolve_node(args.node)
    out_counts: dict[str, int] = {}
    in_counts: dict[str, int] = {}
    for e in idx.out_edges.get(node["id"], []):
        out_counts[e["type"]] = out_counts.get(e["type"], 0) + 1
    for e in idx.in_edges.get(node["id"], []):
        in_counts[e["type"]] = in_counts.get(e["type"], 0) + 1

    if args.json:
        return {"node": node, "out_edges": out_counts, "in_edges": in_counts}

    print(format_node_line(node))
    for k, v in node.get("attrs", {}).items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "…"
        print(f"  attrs.{k}: {v_str}")
    if out_counts:
        print("  outgoing edges: " + ", ".join(f"{t}×{c}" for t, c in sorted(out_counts.items())))
    if in_counts:
        print("  incoming edges: " + ", ".join(f"{t}×{c}" for t, c in sorted(in_counts.items())))
    return None


def cmd_neighbors(idx: GraphIndex, args) -> Any:
    start = idx.resolve_node(args.node)
    visited = {start["id"]: 0}
    frontier = deque([start["id"]])
    hits: list[dict] = []  # {node, via_edge_type, direction, hop}

    while frontier:
        current_id = frontier.popleft()
        depth = visited[current_id]
        if depth >= args.hops:
            continue
        for e, direction in idx.edges_for(current_id, args.dir, args.type):
            neighbor_id = e["target"] if direction == "out" else e["source"]
            if neighbor_id in visited:
                continue
            visited[neighbor_id] = depth + 1
            hits.append({
                "node": idx.nodes.get(neighbor_id, {"id": neighbor_id, "type": "?", "label": neighbor_id, "source": {}}),
                "edge_type": e["type"],
                "direction": direction,
                "hop": depth + 1,
            })
            frontier.append(neighbor_id)

    if args.json:
        return hits
    if not hits:
        print(f"no neighbors found for {start['id']} (dir={args.dir}, type={args.type or 'any'}, hops={args.hops})")
        return None
    for h in sorted(hits, key=lambda h: (h["hop"], h["node"]["id"])):
        arrow = "->" if h["direction"] == "out" else "<-"
        print(f'  [hop {h["hop"]}] {arrow} {h["edge_type"]} {arrow} {format_node_line(h["node"])}')
    return None


def cmd_documents(idx: GraphIndex, args) -> Any:
    """Friendly preset over `neighbors`: both directions of `documents` edges,
    so it works the same whether you point it at a design_doc or a piece of
    infra it might mention."""
    args.type = "documents"
    args.dir = "both"
    args.hops = 1
    return cmd_neighbors(idx, args)


def cmd_path(idx: GraphIndex, args) -> Any:
    start = idx.resolve_node(args.frm)
    end = idx.resolve_node(args.to)
    if start["id"] == end["id"]:
        if args.json:
            return {"path": [start]}
        print(format_node_line(start))
        return None

    # BFS over an undirected view — "how are A and B related" doesn't care
    # which way an edge points, only whether one connects them.
    parent: dict[str, tuple[str, dict, str]] = {}  # node_id -> (prev_id, edge, direction)
    visited = {start["id"]}
    frontier = deque([start["id"]])
    found = False
    while frontier and not found:
        current_id = frontier.popleft()
        for e, direction in idx.edges_for(current_id, "both", None):
            neighbor_id = e["target"] if direction == "out" else e["source"]
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            parent[neighbor_id] = (current_id, e, direction)
            if neighbor_id == end["id"]:
                found = True
                break
            frontier.append(neighbor_id)

    if not found:
        if args.json:
            return {"path": None}
        print(f"no path found between {start['id']} and {end['id']}")
        return None

    chain: list[tuple[str, Optional[dict], Optional[str]]] = []
    node_id = end["id"]
    while node_id != start["id"]:
        prev_id, edge, direction = parent[node_id]
        chain.append((node_id, edge, direction))
        node_id = prev_id
    chain.append((start["id"], None, None))
    chain.reverse()

    if args.json:
        return {"path": [
            {"node": idx.nodes[nid], "via_edge_type": e["type"] if e else None, "direction": d}
            for nid, e, d in chain
        ]}

    for i, (nid, edge, direction) in enumerate(chain):
        node = idx.nodes[nid]
        if i == 0:
            print(format_node_line(node))
        else:
            arrow = "->" if direction == "out" else "<-"
            print(f"  {arrow} {edge['type']} {arrow} {format_node_line(node)}")
    return None


# ── CLI wiring ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    # --graph and --json are declared on a shared `parent` parser and attached
    # to both the top-level parser and every subparser. Plain argparse only
    # recognizes a parent's optional args *before* the subcommand name
    # ("query.py --graph g.json find x" but not "query.py find x --graph
    # g.json") — attaching them to the subparsers too means both orders work,
    # which matters here since the natural phrasing puts the noun (find x)
    # before the flag.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--graph", type=Path, default=Path("output/graph.json"), help="Path to graph.json (default: output/graph.json)")
    common.add_argument("--json", action="store_true")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common])
    sub = ap.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Search nodes by name/label substring", parents=[common])
    p_find.add_argument("text")
    p_find.set_defaults(func=cmd_find)

    p_show = sub.add_parser("show", help="Full detail for one node", parents=[common])
    p_show.add_argument("node")
    p_show.set_defaults(func=cmd_show)

    p_nb = sub.add_parser("neighbors", help="What's connected to this node", parents=[common])
    p_nb.add_argument("node")
    p_nb.add_argument("--type", default=None, help="Restrict to one edge type (contains/calls_module/references/documents)")
    p_nb.add_argument("--dir", choices=EDGE_DIRECTIONS, default="out", help="Edge direction relative to the node (default: out)")
    p_nb.add_argument("--hops", type=int, default=1, help="How many hops to traverse (default: 1)")
    p_nb.set_defaults(func=cmd_neighbors)

    p_doc = sub.add_parser("documents", help='Shortcut: "documents" edges in both directions for this node', parents=[common])
    p_doc.add_argument("node")
    p_doc.set_defaults(func=cmd_documents)

    p_path = sub.add_parser("path", help="Shortest connection between two nodes (any edge type, either direction)", parents=[common])
    p_path.add_argument("frm", metavar="from")
    p_path.add_argument("to")
    p_path.set_defaults(func=cmd_path)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    try:
        idx = load_graph(args.graph)
        result = args.func(idx, args)
    except QueryError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if getattr(args, "json", False) and result is not None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
