"""
Terraform parser for kb_graph.

Deliberately NOT a full HCL2 grammar parser (no `python-hcl2` dependency). We only
need to answer "what blocks exist and what do they refer to" for a first-pass
graph, and a line-oriented scanner gets us there without pulling in a parser that
can choke on `dynamic` blocks, `for` expressions, or provider-specific functions.

Known limitation: brace counting is naive (doesn't special-case braces inside
string literals). Fine for the real-world files this was built against; flag it
if you hit a file with a `{` or `}` inside a quoted string.
"""

from __future__ import annotations

import re
from pathlib import Path

from graph_model import Edge, Graph, Node, Source, node_id

BLOCK_HEADER_RE = re.compile(
    r'^\s*(resource|module|variable|output)\s+"([^"]+)"(?:\s+"([^"]+)")?\s*\{'
)
SOURCE_ATTR_RE = re.compile(r'^\s*source\s*=\s*"([^"]+)"')
VERSION_ATTR_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"')
REF_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def find_repo_root(path: Path) -> Path | None:
    """Walk up from `path` looking for the nearest ancestor with a `.git` dir."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").is_dir():
            return candidate
    return None


def find_tf_files(repo_root: Path) -> list[Path]:
    return sorted(repo_root.rglob("*.tf"))


def parse_repo(graph: Graph, repo_root: Path, source_root: Path) -> str:
    """Parse every .tf file under `repo_root` into `graph`. Returns the repo
    node's id."""
    repo_name = repo_root.name
    repo_nid = node_id("repo", repo_name)
    graph.add_node(
        Node(
            id=repo_nid,
            type="repo",
            name=repo_name,
            label=repo_name,
            source=Source(kind="file", path=str(repo_root.relative_to(source_root))),
            attrs={"kind": "terraform_repo"},
        )
    )

    # declared_names is scoped per containing directory, not per repo: each
    # directory under a repo is its own flat Terraform root/workspace (e.g.
    # dev/, stg/, prd/ are three independent environments that happen to
    # declare identically-named blocks). A flat repo-wide namespace would
    # both collapse same-named blocks from different environments into one
    # node AND silently resolve a `var.x` reference in dev/ against prd/'s
    # declaration of the same name if prd/ was scanned later — dir_key fixes
    # both by keying declarations (and their lookups) to where they live.
    declared_names: dict[str, dict[str, str]] = {}  # dir_key -> {token: block_id}
    # (source_node_id, dir_key, token, file_rel_path, line_no) — resolved once
    # every file in the repo has been scanned, so declaration order across
    # files doesn't matter.
    pending_refs: list[tuple[str, str, str, str, int]] = []

    tf_files = find_tf_files(repo_root)
    for tf_file in tf_files:
        rel_path = str(tf_file.relative_to(source_root))
        rel_dir = str(tf_file.parent.relative_to(repo_root))
        dir_key = "" if rel_dir == "." else rel_dir
        names_here = declared_names.setdefault(dir_key, {})
        file_nid = node_id("file", repo_name, str(tf_file.relative_to(repo_root)))
        graph.add_node(
            Node(
                id=file_nid,
                type="file",
                name=tf_file.name,
                label=rel_path,
                source=Source(kind="file", path=rel_path),
                attrs={"repo": repo_name},
            )
        )
        graph.add_edge(Edge(source=repo_nid, target=file_nid, type="contains"))

        lines = tf_file.read_text(errors="replace").splitlines()
        depth = 0
        current_block: dict | None = None  # {"id", "type", "start_line"}

        for i, line in enumerate(lines, start=1):
            if current_block is None:
                m = BLOCK_HEADER_RE.match(line)
                if m:
                    kind = m.group(1)
                    if kind == "resource":
                        res_type, res_name = m.group(2), m.group(3)
                        block_id = node_id("resource", repo_name, dir_key, res_type, res_name)
                        label = f"{res_type}.{res_name}" + (f"  [{dir_key}]" if dir_key else "")
                        graph.add_node(
                            Node(
                                id=block_id,
                                type="resource",
                                name=f"{res_type}.{res_name}",
                                label=label,
                                source=Source(kind="file", path=rel_path, line=i),
                                attrs={"resource_type": res_type, "repo": repo_name},
                            )
                        )
                        names_here[f"{res_type}.{res_name}"] = block_id
                    else:
                        name = m.group(2)
                        block_id = node_id(
                            "module_call" if kind == "module" else kind, repo_name, dir_key, name
                        )
                        node_type = "module_call" if kind == "module" else kind
                        label = f"{'module' if kind == 'module' else kind}.{name}" + (f"  [{dir_key}]" if dir_key else "")
                        graph.add_node(
                            Node(
                                id=block_id,
                                type=node_type,
                                name=name,
                                label=label,
                                source=Source(kind="file", path=rel_path, line=i),
                                attrs={"repo": repo_name},
                            )
                        )
                        if kind == "module":
                            names_here[f"module.{name}"] = block_id
                        elif kind == "variable":
                            names_here[f"var.{name}"] = block_id

                    graph.add_edge(Edge(source=file_nid, target=block_id, type="contains"))
                    current_block = {"id": block_id, "type": kind}
                    depth += line.count("{") - line.count("}")
                    continue
            else:
                if current_block["type"] == "module":
                    sm = SOURCE_ATTR_RE.match(line)
                    if sm:
                        graph.get_node(current_block["id"]).attrs["module_source"] = sm.group(1)
                    vm = VERSION_ATTR_RE.match(line)
                    if vm:
                        graph.get_node(current_block["id"]).attrs["module_version"] = vm.group(1)

                for pm in REF_TOKEN_RE.finditer(line):
                    token = f"{pm.group(1)}.{pm.group(2)}"
                    pending_refs.append((current_block["id"], dir_key, token, rel_path, i))

                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    current_block = None
                    depth = 0

    for source_id, dir_key, token, rel_path, line_no in pending_refs:
        target_id = declared_names.get(dir_key, {}).get(token)
        if target_id and target_id != source_id:
            graph.add_edge(
                Edge(
                    source=source_id,
                    target=target_id,
                    type="references",
                    attrs={"file": rel_path, "line": line_no},
                )
            )

    return repo_nid
