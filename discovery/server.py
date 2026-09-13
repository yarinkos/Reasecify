#!/usr/bin/env python3
"""
kb_graph discovery UI server.

Stdlib-only local web server (http.server, no new dependency, matching every
other file in kb_graph): serves one self-contained HTML page at `/` and a
POST /api/discover JSON endpoint backed by discovery/harness.py.

Usage:
    export GITHUB_PAT=...
    export GITHUB_HOST=git.example.com    # omit for public github.com
    python discovery/server.py --port 8765
    open http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness import DiscoveryError, run_discovery

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kb_graph discovery</title>
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
    --accent: #2a78d6;
    --bad: #c0392b;
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
      --accent: #3987e5;
      --bad: #e67567;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: var(--page); color: var(--text-primary);
    font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  #app { max-width: 780px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--text-muted); font-size: 12px; margin-bottom: 20px; }
  form {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; display: grid; gap: 10px;
  }
  label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--text-secondary); margin-bottom: 3px; }
  input {
    width: 100%; font: inherit; padding: 7px 9px; border-radius: 5px;
    border: 1px solid var(--border); background: var(--page); color: var(--text-primary);
  }
  .row { display: grid; grid-template-columns: 1fr 140px; gap: 10px; }
  button {
    font: inherit; font-weight: 600; background: var(--accent); color: #fff;
    border: none; border-radius: 5px; padding: 9px 14px; cursor: pointer; justify-self: start;
  }
  button:disabled { opacity: .5; cursor: default; }
  #status { margin: 14px 0; color: var(--text-secondary); font-size: 12px; min-height: 16px; }
  #status.error { color: var(--bad); }
  .card {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; margin-bottom: 10px;
  }
  .card h3 { margin: 0 0 4px; font-size: 14px; }
  .card h3 a { color: var(--text-primary); text-decoration: none; }
  .card h3 a:hover { text-decoration: underline; }
  .badge {
    display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: .03em;
    color: var(--text-muted); border: 1px solid var(--border); border-radius: 3px;
    padding: 1px 5px; margin-left: 6px; vertical-align: middle;
  }
  .desc { color: var(--text-secondary); margin: 4px 0; }
  .why { color: var(--text-muted); font-size: 11px; }
  .why b { color: var(--text-secondary); font-weight: 600; }
  #summary { color: var(--text-muted); font-size: 11px; margin: 16px 0 8px; }
  details { margin-top: 16px; color: var(--text-muted); font-size: 12px; }
  details summary { cursor: pointer; }
</style>
</head>
<body>
<div id="app">
  <h1>kb_graph discovery</h1>
  <div class="sub">Chase a topic from a root repo's Terraform modules into the rest of its GitHub org.</div>
  <form id="f">
    <div>
      <label for="root_repo">Root repo (local path)</label>
      <input id="root_repo" name="root_repo" placeholder="/path/to/already-cloned/repo" required>
    </div>
    <div>
      <label>Discovery mode</label>
      <div style="display:flex; gap:16px; align-items:center; padding-top:2px;">
        <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;color:var(--text-secondary);font-size:13px;">
          <input type="radio" name="mode" value="topic" id="modeTopic" checked style="width:auto;">
          By topic
        </label>
        <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;color:var(--text-secondary);font-size:13px;">
          <input type="radio" name="mode" value="all" id="modeAll" style="width:auto;">
          All referenced repos
        </label>
      </div>
    </div>
    <div id="topicField">
      <label for="topic">Topic</label>
      <input id="topic" name="topic" placeholder="e.g. proxy farm stickiness" required>
    </div>
    <div class="row">
      <div>
        <label for="org">Org override (optional — else read from .git/config)</label>
        <input id="org" name="org" placeholder="leave blank to auto-detect">
      </div>
      <div>
        <label for="max_repos">Max repos</label>
        <input id="max_repos" name="max_repos" type="number" min="1" value="10">
      </div>
    </div>
    <div class="row" style="grid-template-columns: 1fr;">
      <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;color:var(--text-secondary);">
        <input type="checkbox" id="debugToggle" style="width:auto;">
        Show all scanned modules (debug)
      </label>
    </div>
    <button id="submit" type="submit">Discover</button>
  </form>
  <div id="status"></div>
  <div id="summary"></div>
  <div id="results"></div>
  <div id="unresolved"></div>
  <div id="debugModules"></div>
</div>
<script>
(function () {
  "use strict";
  const form = document.getElementById("f");
  const statusEl = document.getElementById("status");
  const summaryEl = document.getElementById("summary");
  const resultsEl = document.getElementById("results");
  const unresolvedEl = document.getElementById("unresolved");
  const debugEl = document.getElementById("debugModules");
  const debugToggle = document.getElementById("debugToggle");
  const submitBtn = document.getElementById("submit");
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const topicField = document.getElementById("topicField");
  const topicInput = document.getElementById("topic");

  function currentMode() {
    return document.querySelector('input[name="mode"]:checked').value;
  }

  // "all" mode resolves every module regardless of topic, so the topic
  // field is meaningless (and not required) in that mode — hide it rather
  // than leave a required input the user has no reason to fill in.
  function applyModeVisibility() {
    const isAll = currentMode() === "all";
    topicField.style.display = isAll ? "none" : "";
    topicInput.required = !isAll;
  }
  modeRadios.forEach(r => r.addEventListener("change", applyModeVisibility));
  applyModeVisibility();

  function applyDebugVisibility() {
    debugEl.style.display = debugToggle.checked ? "" : "none";
  }
  debugToggle.addEventListener("change", applyDebugVisibility);
  applyDebugVisibility();

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function renderResults(data) {
    const isAll = data.mode === "all";
    summaryEl.textContent = isAll
      ? `org: ${data.org} · ${data.org_repo_count} repos in org · ` +
        `${data.modules_considered} modules in root repo · all referenced (no topic filter)`
      : `org: ${data.org} · ${data.org_repo_count} repos in org · ` +
        `${data.modules_considered} modules in root repo · ${data.modules_relevant} relevant to topic`;

    if (!data.matched.length) {
      resultsEl.innerHTML = '<div class="card">' +
        (isAll ? "No modules in this repo resolved to a repo in this org."
                : "No relevant modules resolved to a repo in this org.") +
        '</div>';
    } else {
      resultsEl.innerHTML = data.matched.map(m => `
        <div class="card">
          <h3><a href="${esc(m.html_url)}" target="_blank" rel="noopener">${esc(m.repo)}</a>
            <span class="badge">${m.private ? "private" : "public"}</span>
            ${isAll ? "" : `<span class="badge">score ${m.relevance_score}</span>`}
          </h3>
          ${m.description ? `<div class="desc">${esc(m.description)}</div>` : ""}
          ${m.matched_modules.map(mm => `
            <div class="why"><b>via module</b> ${esc(mm.label)} (<code>${esc(mm.module_source)}</code>)${
              isAll ? "" : ` — shared words: ${esc(mm.topic_matched_words.join(", "))}`
            }</div>
          `).join("")}
        </div>
      `).join("");
    }

    if (data.unresolved_modules.length) {
      unresolvedEl.innerHTML = `<h2 style="font-size:13px;margin:18px 0 6px;">${data.unresolved_modules.length} module(s)${isAll ? "" : " found"}, but no repo in this org matched their source</h2>` +
        data.unresolved_modules.map(m => `<div class="card"><div class="why"><b>${esc(m.label)}</b> — <code>${esc(m.module_source)}</code>${
          isAll ? "" : ` <span class="badge">score ${m.relevance_score}</span> — shared words: ${esc(m.topic_matched_words.join(", "))}`
        }</div></div>`).join("");
    } else {
      unresolvedEl.innerHTML = "";
    }
  }

  function renderDebug(data) {
    if (!data.all_modules || !data.all_modules.length) { debugEl.innerHTML = ""; return; }
    debugEl.innerHTML = `<h2 style="font-size:13px;margin:18px 0 6px;">${data.all_modules.length} module(s) scanned in root repo (every module_call block found, relevant or not)</h2>` +
      '<div class="card" style="padding:0;">' +
      data.all_modules.map((m, i) => `
        <div style="padding:10px 14px;${i ? "border-top:1px solid var(--border);" : ""}${m.relevance_score ? "" : "opacity:.5;"}">
          <div><b>${esc(m.label)}</b> <span class="badge">score ${m.relevance_score}</span></div>
          <div class="why"><code>${esc(m.module_source)}</code></div>
          <div class="why">${m.topic_matched_words.length ? "shared words: " + esc(m.topic_matched_words.join(", ")) : "no shared words with topic — not relevant, never resolved against the org's repos"}</div>
        </div>
      `).join("") +
      '</div>';
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    statusEl.className = "";
    statusEl.textContent = "searching...";
    summaryEl.textContent = "";
    resultsEl.innerHTML = "";
    unresolvedEl.innerHTML = "";
    debugEl.innerHTML = "";
    submitBtn.disabled = true;

    const mode = currentMode();
    const payload = {
      root_repo: document.getElementById("root_repo").value.trim(),
      mode,
      topic: mode === "topic" ? document.getElementById("topic").value.trim() : null,
      org: document.getElementById("org").value.trim() || null,
      max_repos: Number(document.getElementById("max_repos").value) || 10,
    };

    try {
      const res = await fetch("/api/discover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        statusEl.className = "error";
        statusEl.textContent = data.error || `request failed (HTTP ${res.status})`;
        return;
      }
      statusEl.textContent = "";
      renderResults(data);
      renderDebug(data);
    } catch (err) {
      statusEl.className = "error";
      statusEl.textContent = "request failed: " + err.message;
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
</script>
</body>
</html>
"""


class DiscoveryHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's naming)
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/discover":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        root_repo = payload.get("root_repo")
        topic = payload.get("topic") or None
        org = payload.get("org") or None
        mode = payload.get("mode") or "topic"
        try:
            max_repos = int(payload.get("max_repos", 10))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "max_repos must be a number"})
            return

        if mode not in ("topic", "all"):
            self._send_json(400, {"error": f"invalid mode: {mode!r} — expected 'topic' or 'all'"})
            return
        if not root_repo:
            self._send_json(400, {"error": "root_repo is required"})
            return
        if mode == "topic" and not topic:
            self._send_json(400, {"error": "topic is required in 'topic' mode"})
            return

        try:
            result = run_discovery(Path(root_repo), topic, max_repos, org, mode=mode)
        except DiscoveryError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:  # last-resort guard — surface, never crash the server
            self._send_json(500, {"error": f"unexpected error: {e}"})
        else:
            self._send_json(200, result)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        print(f"[discovery] {self.address_string()} - {fmt % args}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = ThreadingHTTPServer(("localhost", args.port), DiscoveryHandler)
    print(f"discovery UI: http://localhost:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
