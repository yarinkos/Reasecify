"""
Confluence page parser for kb_graph.

Handles JSON exports fetched via the Confluence REST API
(`/wiki/rest/api/content/{id}?expand=body.storage,body.view,title,version`), like
the one saved to tmp-data/raw/. Detection is by shape (a `body` key containing
`storage` or `view`), not by filename, so this keeps working if more pages get
dropped into the same source folder later.
"""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

from graph_model import Edge, Graph, Node, Source, node_id

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def looks_like_confluence_export(data: dict) -> bool:
    body = data.get("body")
    if not isinstance(body, dict):
        return False
    return "storage" in body or "view" in body


def html_to_text(html: str) -> str:
    text = TAG_RE.sub(" ", html)
    text = unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_confluence_file(graph: Graph, json_path: Path, source_root: Path) -> str | None:
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if not looks_like_confluence_export(data):
        return None

    page_id = str(data.get("id", json_path.stem))
    title = data.get("title", json_path.stem)

    body = data.get("body", {})
    html = (body.get("view") or body.get("storage") or {}).get("value", "")
    text = html_to_text(html)

    links = data.get("_links", {})
    url = None
    if links.get("base") and links.get("webui"):
        url = links["base"] + links["webui"]

    doc_nid = node_id("design_doc", page_id)
    graph.add_node(
        Node(
            id=doc_nid,
            type="design_doc",
            name=title,
            label=title,
            source=Source(kind="confluence", page_id=page_id, url=url),
            attrs={
                "text": text,
                "version": (data.get("version") or {}).get("number"),
                "local_file": str(json_path.relative_to(source_root)),
            },
        )
    )
    return doc_nid
