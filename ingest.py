#!/usr/bin/env python3
"""
kb_graph ingest CLI.

Walks a `--source` folder, dispatches each file to the Terraform or Confluence
parser by shape (not by hardcoded subfolder names — see parsers/), links the
results with two cheap heuristics (module `source` string containment for
repo-to-repo `calls_module` edges, keyword overlap for design_doc `documents`
edges), and writes three outputs under `--out`:

  graph.json          the full typed node/edge graph, the contract other
                       tooling (a future query/lint step) can build on
  GRAPH_SUMMARY.md     human-readable counts + the documents edges found,
                       for sanity-checking the doc-linking heuristic
  graph.html           a single self-contained viewer (vanilla JS + canvas,
                       no CDN, no build step, data inlined at generation time)

No LLM calls, no embeddings — this is a first pass to validate the pipeline
shape, not a finished linker. See README.md for known limitations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from graph_model import Edge, Graph, Node, Source, node_id
from parsers.confluence import looks_like_confluence_export, parse_confluence_file
from parsers.terraform import find_repo_root, parse_repo

# Node types that are plausibly "infra" a design doc might describe. Deliberately
# excludes `file` (source-file names are mostly noise: "main", "outputs") and
# `design_doc` itself.
DOCUMENTABLE_TYPES = {"repo", "resource", "module_call", "variable", "output", "external_module"}

WORD_RE = re.compile(r"[A-Za-z0-9]+")
MIN_KEYWORD_LEN = 5  # below this, words are too generic ("main", "var") to be signal


def discover_repo_roots(source_root: Path) -> list[Path]:
    """Every distinct git-repo root that owns at least one .tf file under
    source_root, found by walking up from each .tf file to its nearest .git
    ancestor. Repos with no .git ancestor inside source_root are skipped with a
    warning rather than silently mis-rooted."""
    roots: set[Path] = set()
    for tf_file in sorted(source_root.rglob("*.tf")):
        root = find_repo_root(tf_file)
        if root is None:
            print(f"warning: no .git ancestor for {tf_file}, skipping", file=sys.stderr)
            continue
        try:
            root.relative_to(source_root)
        except ValueError:
            print(f"warning: repo root {root} for {tf_file} is outside --source, skipping", file=sys.stderr)
            continue
        roots.add(root)
    return sorted(roots)


def discover_confluence_files(source_root: Path) -> list[Path]:
    candidates = []
    for json_file in sorted(source_root.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and looks_like_confluence_export(data):
            candidates.append(json_file)
    return candidates


# Words too generic to count as a signal that a module source names a specific
# repo (vendor/cloud/repo-type boilerplate that shows up in nearly every repo
# and registry path alike).
GENERIC_REPO_WORDS = {"tf", "terraform", "google", "aws", "azure", "gcp", "module", "modules"}

WORD_SPLIT_RE = re.compile(r"[/\-_.]+")
# A pure hex/UUID-ish path segment (env0-style private-registry paths embed
# one) — never a real signal, and never picked as an external_module label.
OPAQUE_SEGMENT_RE = re.compile(r"^[0-9a-f-]+$")
# A registry hostname segment (e.g. "api.env0.com") — repo/module name segments
# in these source strings use hyphens/underscores, never a literal dot, so
# "contains a dot" reliably picks out the host component instead. Without this,
# a short module name (few significant words) can lose to the host segment,
# which decomposes into several short "words" (api/env0/com) that pass the
# significant_words length filter on their own.
HOST_SEGMENT_RE = re.compile(r"\.")


def significant_words(name: str) -> set[str]:
    return {
        w for w in WORD_SPLIT_RE.split(name.lower())
        if w and len(w) > 2 and w not in GENERIC_REPO_WORDS
    }


def external_module_label(source: str) -> str:
    """Best-effort human label for a module source that didn't resolve to any
    cloned repo — picks the "/"-separated segment that looks most like a repo
    name (most significant words), skipping registry hosts and opaque
    hash/UUID segments. Falls back to the raw source if nothing looks better."""
    segments = [
        s for s in source.split("/")
        if s and not OPAQUE_SEGMENT_RE.match(s.lower()) and not HOST_SEGMENT_RE.search(s)
    ]
    if not segments:
        return source
    return max(segments, key=lambda s: (len(significant_words(s)), len(s)))


def link_calls_module(graph: Graph, repo_names: list[str]) -> int:
    """For every module_call node with a `module_source` attr, match against
    known repo directory names by significant-word overlap rather than plain
    substring containment: a private-registry source string like
    ".../foundations-network-gateway-compute-module/google" doesn't literally
    contain the cloned repo's directory name
    ("tf-google-foundations-network-gateway-compute-module") — the
    "tf-google-" vendor prefix isn't in the registry path — but after
    dropping generic vendor/cloud words, both sides share the same
    distinctive words ("foundations", "network", "gateway", "compute").

    Matching requires *full* containment — every one of the repo's significant
    words must appear in the source — not just some fixed-size overlap. A
    fixed-count threshold (e.g. "3+ shared words") produced a false positive:
    a *different*, uncloned module ("...-load-balancer-module") shared enough
    generic-adjacent words ("foundations", "network", "gateway") with the one
    repo that happened to be cloned to clear the threshold, even though
    "compute" (part of the cloned repo's name) never appeared in that source
    at all, and "load"/"balancer" (part of the real target's name) never
    appeared in the cloned repo's name either — a real mismatch on both
    sides, not just imprecise naming.

    When no cloned repo's words are fully contained in the source, the call
    links to a synthesized `external_module` node instead of guessing or
    dropping the edge — the dependency is real and worth surfacing even when
    its target isn't in this corpus. Not real Terraform module resolution,
    just enough for a first pass."""
    count = 0
    repo_word_sets = {name: significant_words(name) for name in repo_names}
    for n in graph.nodes:
        if n.type != "module_call":
            continue
        source = n.attrs.get("module_source")
        if not source:
            continue
        source_words = significant_words(source)
        matched_repo = next(
            (name for name, words in repo_word_sets.items() if words and words <= source_words),
            None,
        )
        if matched_repo:
            repo_nid = node_id("repo", matched_repo)
            if repo_nid != n.id and graph.has_node(repo_nid):
                graph.add_edge(
                    Edge(source=n.id, target=repo_nid, type="calls_module",
                         attrs={"matched_words": sorted(repo_word_sets[matched_repo])})
                )
                count += 1
        else:
            ext_nid = node_id("external_module", source)
            if not graph.has_node(ext_nid):
                label = external_module_label(source)
                graph.add_node(
                    Node(
                        id=ext_nid,
                        type="external_module",
                        name=label,
                        label=label,
                        source=Source(kind="external"),
                        attrs={"module_source": source, "note": "not present in this ingest's source folder"},
                    )
                )
            graph.add_edge(
                Edge(source=n.id, target=ext_nid, type="calls_module", attrs={"resolved": False})
            )
            count += 1
    return count


def extract_words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) >= MIN_KEYWORD_LEN and not w.isdigit()}


# A single shared word this short or longer is distinctive enough on its own
# ("web_service", "foundations", "stickiness"). Below that, a lone match is more
# likely boilerplate this whole GCP/Terraform corpus shares ("google",
# "module", "compute") and needs a second matching word to count as signal.
MIN_SOLO_KEYWORD_LEN = 8
MIN_KEYWORD_HITS = 2


def link_documents(graph: Graph) -> list[Edge]:
    """design_doc -> infra node keyword-overlap linking. No ranking or full
    disambiguation (see README known limitations) — this only tightens the
    obvious noise case where a single short, generic shared word (every GCP
    resource says "google") would otherwise count as a hit on its own."""
    created: list[Edge] = []
    doc_nodes = [n for n in graph.nodes if n.type == "design_doc"]
    candidates = [n for n in graph.nodes if n.type in DOCUMENTABLE_TYPES]
    for doc in doc_nodes:
        doc_words = extract_words(doc.attrs.get("text", ""))
        if not doc_words:
            continue
        for cand in candidates:
            cand_words = extract_words(cand.name) | extract_words(cand.label)
            matched = sorted(doc_words & cand_words)
            is_signal = len(matched) >= MIN_KEYWORD_HITS or (
                len(matched) == 1 and len(matched[0]) >= MIN_SOLO_KEYWORD_LEN
            )
            if matched and is_signal:
                edge = Edge(
                    source=doc.id,
                    target=cand.id,
                    type="documents",
                    attrs={"keywords": matched},
                )
                graph.add_edge(edge)
                created.append(edge)
    return created


def write_graph_json(graph: Graph, out_dir: Path, meta: dict) -> Path:
    path = out_dir / "graph.json"
    path.write_text(json.dumps(graph.to_dict(meta), indent=2))
    return path


def write_summary(graph: Graph, out_dir: Path, documents_edges: list[Edge], meta: dict) -> Path:
    node_counts = graph.counts_by_type()
    edge_counts: dict[str, int] = {}
    for e in graph.edges:
        edge_counts[e.type] = edge_counts.get(e.type, 0) + 1

    lines = ["# kb_graph summary", ""]
    lines.append(f"Source: `{meta['source']}`")
    lines.append("")
    lines.append("## Nodes")
    for t in sorted(node_counts):
        lines.append(f"- {t}: {node_counts[t]}")
    lines.append(f"- **total**: {len(graph.nodes)}")
    lines.append("")
    lines.append("## Edges")
    for t in sorted(edge_counts):
        lines.append(f"- {t}: {edge_counts[t]}")
    lines.append(f"- **total**: {len(graph.edges)}")
    lines.append("")
    lines.append("## `documents` edges (design_doc → infra, keyword overlap)")
    if not documents_edges:
        lines.append("_none found_")
    else:
        by_doc: dict[str, list[Edge]] = {}
        for e in documents_edges:
            by_doc.setdefault(e.source, []).append(e)
        for doc_id, edges in by_doc.items():
            doc_node = graph.get_node(doc_id)
            lines.append(f"- **{doc_node.label if doc_node else doc_id}** ({len(edges)} links):")
            for e in sorted(edges, key=lambda e: e.target):
                target = graph.get_node(e.target)
                keywords = ", ".join(e.attrs.get("keywords", []))
                lines.append(f"  - {target.label if target else e.target} — via: {keywords}")
    lines.append("")

    path = out_dir / "GRAPH_SUMMARY.md"
    path.write_text("\n".join(lines))
    return path


# ── graph.html rendering ─────────────────────────────────────────────────────
#
# Categorical palette per the dataviz skill's validated default (palette.md),
# slots 1-7 in fixed order. Validated with scripts/validate_palette.js:
#   - adjacent pairs: PASS in both light and dark (this is the mode the
#     skill's default order is validated for)
#   - all-pairs (the honest mode for a node-link graph, where any two node
#     types can sit side by side on screen): FAILS past 3 slots, same as the
#     skill documents for scatter/bubble/choropleth. Since we have 7 types
#     to distinguish, color here is a SECONDARY channel — each type also gets
#     a distinct marker shape, so identity never depends on color alone.
NODE_STYLE = {
    "repo":        {"light": "#2a78d6", "dark": "#3987e5", "shape": "circle"},
    "file":        {"light": "#eb6834", "dark": "#d95926", "shape": "square"},
    "resource":    {"light": "#1baf7a", "dark": "#199e70", "shape": "triangle"},
    "module_call": {"light": "#eda100", "dark": "#c98500", "shape": "diamond"},
    "variable":    {"light": "#e87ba4", "dark": "#d55181", "shape": "pentagon"},
    "output":      {"light": "#008300", "dark": "#008300", "shape": "plus"},
    "design_doc":  {"light": "#4a3aa7", "dark": "#9085e9", "shape": "star"},
    "external_module": {"light": "#e34948", "dark": "#e66767", "shape": "hexagon"},
}

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kb_graph viewer</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    __LIGHT_VARS__
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      __DARK_VARS__
    }
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%;
    background: var(--page);
    color: var(--text-primary);
    font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  #app { display: flex; height: 100vh; }
  #sidebar {
    width: 260px; flex: 0 0 260px; overflow-y: auto;
    background: var(--surface-1); border-right: 1px solid var(--border);
    padding: 14px;
  }
  #sidebar h1 { font-size: 14px; margin: 0 0 4px; }
  #sidebar .meta { color: var(--text-muted); font-size: 11px; margin-bottom: 14px; }
  #sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--text-secondary); margin: 16px 0 6px; }
  .legend-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; user-select: none; }
  .legend-row.off { opacity: .35; }
  .legend-swatch { width: 16px; height: 16px; flex: 0 0 16px; }
  .legend-count { margin-left: auto; color: var(--text-muted); font-variant-numeric: tabular-nums; }
  .legend-label { color: var(--text-primary); }
  button {
    font: inherit; background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 4px; padding: 5px 10px; cursor: pointer;
  }
  button:hover { background: var(--gridline); }
  #controls { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  #canvasWrap { position: relative; flex: 1; overflow: hidden; }
  canvas { display: block; background: var(--surface-1); cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #tooltip {
    position: absolute; pointer-events: none; z-index: 5;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 8px; font-size: 12px; max-width: 320px; display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,.15);
  }
  #tooltip .t-type { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: .03em; }
  #tooltip .t-label { color: var(--text-primary); font-weight: 600; margin: 2px 0; }
  #tooltip .t-source { color: var(--text-secondary); font-size: 11px; word-break: break-all; }
  #tableView {
    position: absolute; inset: 0; background: var(--surface-1); overflow: auto;
    padding: 16px; display: none; z-index: 4;
  }
  table { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--gridline); }
  th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
  td.mono { font-family: ui-monospace, monospace; color: var(--text-secondary); }
  #hint { color: var(--text-muted); font-size: 11px; margin-top: 14px; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>kb_graph</h1>
    <div class="meta">__META_LINE__</div>
    <h2>Node types (click to toggle)</h2>
    <div id="legend"></div>
    <h2>View</h2>
    <div id="controls">
      <button id="btnReset">Reset view</button>
      <button id="btnTable">Table view</button>
    </div>
    <div id="hint">Drag background to pan · scroll to zoom · drag a node to move it · hover for details.</div>
  </div>
  <div id="canvasWrap">
    <canvas id="canvas"></canvas>
    <div id="tooltip"></div>
    <div id="tableView"></div>
  </div>
</div>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("graph-data").textContent);
  const STYLE = __STYLE_JSON__;
  const isDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  const nodes = DATA.nodes.map((n, i) => ({
    ...n,
    x: (Math.cos(i * 2.399963) * (40 + i * 3)),
    y: (Math.sin(i * 2.399963) * (40 + i * 3)),
    vx: 0, vy: 0,
  }));
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  const edges = DATA.edges
    .map(e => ({ ...e, a: nodeById.get(e.source), b: nodeById.get(e.target) }))
    .filter(e => e.a && e.b);

  const typeVisible = {};
  for (const t of Object.keys(STYLE)) typeVisible[t] = true;

  // ── canvas + camera ─────────────────────────────────────────────────────
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const wrap = document.getElementById("canvasWrap");
  let camX = 0, camY = 0, camScale = 1;
  // One shared zoom range for both fitView()'s auto-fit and manual wheel-zoom
  // — previously fitView() allowed down to 0.05 but the wheel handler
  // clamped at 0.15, so on a widely-spread graph the very first zoom-out
  // scroll after an auto-fit would suddenly snap *in* to 0.15, a jarring
  // jump that reads as nodes "moving around" on their own.
  const CAM_SCALE_MIN = 0.05, CAM_SCALE_MAX = 6;

  function resize() {
    const r = wrap.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    canvas.style.width = r.width + "px";
    canvas.style.height = r.height + "px";
  }
  window.addEventListener("resize", resize);
  resize();

  function worldToScreen(x, y) {
    const r = wrap.getBoundingClientRect();
    return [
      (x - camX) * camScale + r.width / 2,
      (y - camY) * camScale + r.height / 2,
    ];
  }
  function screenToWorld(sx, sy) {
    const r = wrap.getBoundingClientRect();
    return [
      (sx - r.width / 2) / camScale + camX,
      (sy - r.height / 2) / camScale + camY,
    ];
  }

  // ── force-directed layout (simple, damped, cools down and stops) ─────────
  const REPULSION = 2600;
  const SPRING_LEN = 70;
  const SPRING_K = 0.02;
  const DAMPING = 0.85;
  const CENTER_K = 0.002;

  // Previously this ran forever at full strength: with no cooling, a graph
  // that hadn't fully converged (bigger graphs take longer, since pairwise
  // repulsion's equilibrium spread grows with node count) just kept drifting
  // — visible as nodes perpetually "moving around". And since the one-time
  // auto-fit below used to fire at a fixed frame count rather than once the
  // layout actually settled, nodes that were still drifting past that point
  // could end up outside the frozen camera and look like they'd vanished.
  // `alpha` fixes both: it starts at 1 and decays toward 0 (same schedule
  // d3-force defaults to), scaling how much new force can still move a node
  // each frame. Once it crosses ALPHA_MIN the layout is considered settled —
  // step() stops being called (see loop()) and positions freeze — and *that*
  // is when the camera fits itself to the real final layout, not a guess.
  let alpha = 1;
  const ALPHA_DECAY = 0.0228;
  const ALPHA_MIN = 0.001;

  function step() {
    for (const n of nodes) { n.fx = 0; n.fy = 0; }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        let d = Math.sqrt(d2);
        const f = REPULSION / d2;
        dx /= d; dy /= d;
        a.fx += dx * f; a.fy += dy * f;
        b.fx -= dx * f; b.fy -= dy * f;
      }
    }
    for (const e of edges) {
      let dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
      let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - SPRING_LEN) * SPRING_K;
      dx /= d; dy /= d;
      e.a.fx += dx * f; e.a.fy += dy * f;
      e.b.fx -= dx * f; e.b.fy -= dy * f;
    }
    for (const n of nodes) {
      if (n === dragNode) continue;
      n.fx -= n.x * CENTER_K;
      n.fy -= n.y * CENTER_K;
      // Scale the newly-computed force by alpha (not the existing velocity —
      // DAMPING alone still shrinks that every frame) so movement fades out
      // smoothly as the layout cools instead of stopping abruptly.
      n.vx = (n.vx + n.fx * alpha) * DAMPING;
      n.vy = (n.vy + n.fy * alpha) * DAMPING;
      n.x += n.vx;
      n.y += n.vy;
    }
    alpha *= (1 - ALPHA_DECAY);
  }

  // ── shape drawing (identity channel #1 — never rely on color alone) ─────
  // Takes the target context explicitly so the main canvas and the legend
  // swatches (each their own tiny canvas) share one definition.
  function drawShape(g, shape, x, y, r) {
    g.beginPath();
    switch (shape) {
      case "circle":
        g.arc(x, y, r, 0, Math.PI * 2);
        break;
      case "square":
        g.rect(x - r, y - r, r * 2, r * 2);
        break;
      case "triangle":
        g.moveTo(x, y - r);
        g.lineTo(x + r * 0.87, y + r * 0.5);
        g.lineTo(x - r * 0.87, y + r * 0.5);
        g.closePath();
        break;
      case "diamond":
        g.moveTo(x, y - r);
        g.lineTo(x + r, y);
        g.lineTo(x, y + r);
        g.lineTo(x - r, y);
        g.closePath();
        break;
      case "pentagon":
        for (let k = 0; k < 5; k++) {
          const a = -Math.PI / 2 + k * (2 * Math.PI / 5);
          const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
          k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
        }
        g.closePath();
        break;
      case "star":
        for (let k = 0; k < 10; k++) {
          const a = -Math.PI / 2 + k * (Math.PI / 5);
          const rr = k % 2 === 0 ? r : r * 0.45;
          const px = x + rr * Math.cos(a), py = y + rr * Math.sin(a);
          k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
        }
        g.closePath();
        break;
      case "plus": {
        const w = r * 0.4;
        g.rect(x - w, y - r, w * 2, r * 2);
        g.rect(x - r, y - w, r * 2, w * 2);
        break;
      }
      case "hexagon":
        for (let k = 0; k < 6; k++) {
          const a = -Math.PI / 2 + k * (Math.PI / 3);
          const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
          k === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
        }
        g.closePath();
        break;
      default:
        g.arc(x, y, r, 0, Math.PI * 2);
    }
  }

  function render() {
    const dark = isDark();
    const cs = getComputedStyle(document.documentElement);
    const surface = cs.getPropertyValue("--surface-1").trim();
    const gridline = cs.getPropertyValue("--gridline").trim();
    const textSecondary = cs.getPropertyValue("--text-secondary").trim();

    ctx.save();
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    const r = wrap.getBoundingClientRect();
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, r.width, r.height);

    // edges
    ctx.lineWidth = 1;
    for (const e of edges) {
      if (!typeVisible[e.a.type] || !typeVisible[e.b.type]) continue;
      const [ax, ay] = worldToScreen(e.a.x, e.a.y);
      const [bx, by] = worldToScreen(e.b.x, e.b.y);
      ctx.strokeStyle = e.type === "documents" ? textSecondary : gridline;
      ctx.globalAlpha = e.type === "documents" ? 0.55 : 0.8;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // nodes
    for (const n of nodes) {
      if (!typeVisible[n.type]) continue;
      const style = STYLE[n.type] || { light: "#999", dark: "#999", shape: "circle" };
      const [sx, sy] = worldToScreen(n.x, n.y);
      const radius = (hoverNode === n ? 8 : 6) * Math.min(1.4, Math.max(0.6, camScale));
      ctx.fillStyle = dark ? style.dark : style.light;
      ctx.strokeStyle = surface;
      ctx.lineWidth = 1.5;
      drawShape(ctx, style.shape, sx, sy, radius);
      ctx.fill();
      ctx.stroke();
      if (hoverNode === n) {
        ctx.strokeStyle = textSecondary;
        ctx.lineWidth = 1;
        drawShape(ctx, style.shape, sx, sy, radius + 3);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // The force layout's equilibrium spread scales with node count (repulsion is
  // pairwise), so a fixed initial camera cuts off graphs bigger than ~a dozen
  // nodes. Fit the camera to the current node bounding box instead of assuming
  // a size — called once the layout has had time to settle, and again on demand.
  function fitView() {
    if (!nodes.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (!typeVisible[n.type]) continue;
      minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
      minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    }
    if (!isFinite(minX)) return;
    const r = wrap.getBoundingClientRect();
    const spanX = Math.max(1, maxX - minX), spanY = Math.max(1, maxY - minY);
    const margin = 0.85; // leave breathing room around the bounds
    camScale = Math.min(CAM_SCALE_MAX, Math.max(CAM_SCALE_MIN, margin * Math.min(r.width / spanX, r.height / spanY)));
    camX = (minX + maxX) / 2;
    camY = (minY + maxY) / 2;
  }

  let autoFitDone = false, settled = false;

  function loop() {
    if (!settled) {
      step();
      if (alpha < ALPHA_MIN) {
        // Layout has actually converged (not just "150 frames elapsed") —
        // freeze positions (step() stops running) and fit the camera to
        // where nodes really ended up, once, on first settle. A later
        // re-settle (e.g. after a drag reheats alpha, see pointerdown)
        // intentionally does NOT re-trigger this — the user positioned
        // things on purpose, so don't yank their camera around under them.
        settled = true;
        if (!autoFitDone) {
          fitView();
          autoFitDone = true;
        }
      }
    }
    render();
    requestAnimationFrame(loop);
  }

  // ── interaction: pan, zoom, drag, hover (hit target > painted pixels) ───
  let isPanning = false, panStart = null, camStart = null;
  let dragNode = null;
  let hoverNode = null;
  const HIT_RADIUS_SCREEN = 14; // bigger than the painted mark, per interaction.md

  function nearestNode(sx, sy) {
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      if (!typeVisible[n.type]) continue;
      const [nx, ny] = worldToScreen(n.x, n.y);
      const d = Math.hypot(nx - sx, ny - sy);
      if (d < bestD) { bestD = d; best = n; }
    }
    return bestD <= HIT_RADIUS_SCREEN ? best : null;
  }

  const tooltip = document.getElementById("tooltip");
  function showTooltip(n, sx, sy) {
    if (!n) { tooltip.style.display = "none"; return; }
    tooltip.innerHTML = "";
    const t = document.createElement("div"); t.className = "t-type"; t.textContent = n.type;
    const l = document.createElement("div"); l.className = "t-label"; l.textContent = n.label;
    const s = document.createElement("div"); s.className = "t-source";
    const src = n.source || {};
    s.textContent = src.url ? src.url : (src.path ? src.path + (src.line ? ":" + src.line : "") : "");
    tooltip.appendChild(t); tooltip.appendChild(l); if (s.textContent) tooltip.appendChild(s);
    tooltip.style.display = "block";
    tooltip.style.left = Math.min(sx + 14, wrap.clientWidth - 330) + "px";
    tooltip.style.top = Math.max(sy - 10, 4) + "px";
  }

  canvas.addEventListener("pointerdown", (ev) => {
    const r = wrap.getBoundingClientRect();
    const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    const n = nearestNode(sx, sy);
    if (n) {
      dragNode = n;
      // The layout may have long since settled (alpha decayed to ~0, step()
      // no longer running) — reheat it so neighbors ease out of the way of
      // the dragged node instead of staying frozen mid-drag.
      alpha = Math.max(alpha, 0.3);
      settled = false;
    } else {
      isPanning = true;
      panStart = [ev.clientX, ev.clientY];
      camStart = [camX, camY];
      canvas.classList.add("dragging");
    }
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const r = wrap.getBoundingClientRect();
    const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    if (dragNode) {
      const [wx, wy] = screenToWorld(sx, sy);
      dragNode.x = wx; dragNode.y = wy; dragNode.vx = 0; dragNode.vy = 0;
    } else if (isPanning) {
      camX = camStart[0] - (ev.clientX - panStart[0]) / camScale;
      camY = camStart[1] - (ev.clientY - panStart[1]) / camScale;
    } else {
      hoverNode = nearestNode(sx, sy);
      showTooltip(hoverNode, sx, sy);
    }
  });
  function endDrag(ev) {
    dragNode = null;
    isPanning = false;
    canvas.classList.remove("dragging");
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => { hoverNode = null; showTooltip(null); });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.001);
    camScale = Math.min(CAM_SCALE_MAX, Math.max(CAM_SCALE_MIN, camScale * factor));
  }, { passive: false });

  document.getElementById("btnReset").addEventListener("click", () => {
    fitView();
  });

  // ── legend (identity channel #2 — shape+color+label, never color alone) ─
  const counts = {};
  for (const n of nodes) counts[n.type] = (counts[n.type] || 0) + 1;
  const legend = document.getElementById("legend");
  for (const type of Object.keys(STYLE)) {
    if (!(type in counts)) continue;
    const row = document.createElement("div");
    row.className = "legend-row";
    const sw = document.createElement("canvas");
    sw.width = 32; sw.height = 32; sw.className = "legend-swatch";
    const sctx = sw.getContext("2d");
    sctx.fillStyle = isDark() ? STYLE[type].dark : STYLE[type].light;
    drawShape(sctx, STYLE[type].shape, 16, 16, 9);
    sctx.fill();
    const label = document.createElement("span");
    label.className = "legend-label";
    label.textContent = type + " (" + STYLE[type].shape + ")";
    const count = document.createElement("span");
    count.className = "legend-count";
    count.textContent = String(counts[type]);
    row.appendChild(sw); row.appendChild(label); row.appendChild(count);
    row.addEventListener("click", () => {
      typeVisible[type] = !typeVisible[type];
      row.classList.toggle("off", !typeVisible[type]);
    });
    legend.appendChild(row);
  }

  // ── table view (accessibility fallback — every value reachable without hover) ─
  const tableView = document.getElementById("tableView");
  const btnTable = document.getElementById("btnTable");
  let tableShown = false;
  function buildTable() {
    tableView.innerHTML = "";
    const h1 = document.createElement("h2"); h1.textContent = "Nodes"; tableView.appendChild(h1);
    const t1 = document.createElement("table");
    t1.innerHTML = "<thead><tr><th>Type</th><th>Label</th><th>Source</th></tr></thead>";
    const tb1 = document.createElement("tbody");
    for (const n of DATA.nodes) {
      const tr = document.createElement("tr");
      const c1 = document.createElement("td"); c1.textContent = n.type;
      const c2 = document.createElement("td"); c2.textContent = n.label;
      const c3 = document.createElement("td"); c3.className = "mono";
      const src = n.source || {};
      c3.textContent = src.url || (src.path ? src.path + (src.line ? ":" + src.line : "") : "");
      tr.appendChild(c1); tr.appendChild(c2); tr.appendChild(c3);
      tb1.appendChild(tr);
    }
    t1.appendChild(tb1); tableView.appendChild(t1);

    const h2 = document.createElement("h2"); h2.textContent = "Edges"; tableView.appendChild(h2);
    const t2 = document.createElement("table");
    t2.innerHTML = "<thead><tr><th>Type</th><th>From</th><th>To</th></tr></thead>";
    const tb2 = document.createElement("tbody");
    for (const e of DATA.edges) {
      const tr = document.createElement("tr");
      const c1 = document.createElement("td"); c1.textContent = e.type;
      const c2 = document.createElement("td"); c2.textContent = (nodeById.get(e.source) || {}).label || e.source;
      const c3 = document.createElement("td"); c3.textContent = (nodeById.get(e.target) || {}).label || e.target;
      tr.appendChild(c1); tr.appendChild(c2); tr.appendChild(c3);
      tb2.appendChild(tr);
    }
    t2.appendChild(tb2); tableView.appendChild(t2);
  }
  btnTable.addEventListener("click", () => {
    tableShown = !tableShown;
    if (tableShown && !tableView.dataset.built) {
      buildTable();
      tableView.dataset.built = "1";
    }
    tableView.style.display = tableShown ? "block" : "none";
    btnTable.textContent = tableShown ? "Graph view" : "Table view";
  });

  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""


def render_html(graph: Graph, out_dir: Path, meta: dict) -> Path:
    light_vars = "\n    ".join(
        f"--n-{t}: {s['light']};" for t, s in NODE_STYLE.items()
    )
    dark_vars = "\n      ".join(
        f"--n-{t}: {s['dark']};" for t, s in NODE_STYLE.items()
    )
    graph_json = json.dumps(graph.to_dict(meta)).replace("</", "<\\/")
    style_json = json.dumps(NODE_STYLE)
    meta_line = f"{len(graph.nodes)} nodes · {len(graph.edges)} edges · source: {meta['source']}"

    html = (
        HTML_TEMPLATE
        .replace("__LIGHT_VARS__", light_vars)
        .replace("__DARK_VARS__", dark_vars)
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__STYLE_JSON__", style_json)
        .replace("__META_LINE__", meta_line)
    )
    path = out_dir / "graph.html"
    path.write_text(html)
    return path


class IngestError(Exception):
    """User-facing failure (bad source path) — same role as query.py's
    QueryError / discovery's DiscoveryError, so a caller (main(), or
    kb_graph/workflow's server) can catch one thing and report it cleanly
    instead of a stack trace."""


def run_ingest(sources: list[Path], out: Path) -> dict:
    """Programmatic core of the ingest CLI. Same pipeline `main()` has always
    run, just re-entrant over a *list* of source roots instead of one — so a
    caller can point this at several independently-cloned repo directories
    (plus the original root repo that isn't itself "discovered") in a single
    run, without first merging them into one directory or relying on
    symlinks. Each source is discovered/parsed against its own root, exactly
    as `main()` already did for a single `--source`; only the accumulation
    across sources is new.

    Returns the same counts `main()` used to only print, plus the paths of
    the three files written — the contract `kb_graph/workflow`'s
    `/api/ingest` endpoint hands back to the browser."""
    out_dir = out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_sources: list[Path] = []
    for src in sources:
        resolved = src.resolve()
        if not resolved.is_dir():
            raise IngestError(f"source is not a directory: {resolved}")
        resolved_sources.append(resolved)

    graph = Graph()
    repo_names: list[str] = []
    confluence_file_count = 0

    for source_root in resolved_sources:
        repo_roots = discover_repo_roots(source_root)
        for repo_root in repo_roots:
            parse_repo(graph, repo_root, source_root)
        repo_names.extend(r.name for r in repo_roots)

        confluence_files = discover_confluence_files(source_root)
        for cf in confluence_files:
            parse_confluence_file(graph, cf, source_root)
        confluence_file_count += len(confluence_files)

    calls_module_count = link_calls_module(graph, repo_names)
    documents_edges = link_documents(graph)

    meta = {
        "source": ", ".join(str(s) for s in resolved_sources),
        # "sources" (list, machine-readable) alongside the display-only
        # "source" string above — kb_graph/common/context.py needs the
        # individual source roots back (not just a joined string) to
        # resolve a node's source-relative path to an absolute file on
        # disk for chat snippets.
        "sources": [str(s) for s in resolved_sources],
        "repos": repo_names,
        "confluence_pages": confluence_file_count,
    }

    json_path = write_graph_json(graph, out_dir, meta)
    summary_path = write_summary(graph, out_dir, documents_edges, meta)
    html_path = render_html(graph, out_dir, meta)

    return {
        "sources": [str(s) for s in resolved_sources],
        "out": str(out_dir),
        "repos": repo_names,
        "confluence_pages": confluence_file_count,
        "calls_module_edges": calls_module_count,
        "documents_edges": len(documents_edges),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "graph_json": str(json_path),
        "summary_md": str(summary_path),
        "graph_html": str(html_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="Root folder to walk for .tf and Confluence .json files")
    ap.add_argument("--out", required=True, type=Path, help="Output folder for graph.json / GRAPH_SUMMARY.md / graph.html")
    args = ap.parse_args()

    try:
        result = run_ingest([args.source], args.out)
    except IngestError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    repos = result["repos"]
    print(f"repos found:            {len(repos)} ({', '.join(repos) or '-'})")
    print(f"confluence pages found: {result['confluence_pages']}")
    print(f"calls_module edges:     {result['calls_module_edges']}")
    print(f"documents edges:        {result['documents_edges']}")
    print(f"nodes / edges:          {result['nodes']} / {result['edges']}")
    print(f"wrote {result['graph_json']}")
    print(f"wrote {result['summary_md']}")
    print(f"wrote {result['graph_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
