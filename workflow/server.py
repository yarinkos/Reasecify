#!/usr/bin/env python3
"""
kb_graph workflow server — the "connect the dots" page.

`discovery/server.py` stays exactly as it is: a standalone page that only
answers "which repos are relevant." This is a **separate** module that
orchestrates that page's underlying library (`discovery/harness.py`)
together with `ingest.py` and the new `common/` helpers into one guided
sequence: Discover -> pick repos -> Clone -> Ingest (build the graph) ->
Chat (ask questions answered from the graph). It imports those existing
pieces directly (same process, no HTTP hop between the two servers) rather
than duplicating any of their logic.

Runs on its own port (default 8766) so it can sit alongside
`discovery/server.py` (default 8765) rather than replacing it.

Usage:
    export GITHUB_PAT=...                 # required for discover + clone
    export GITHUB_HOST=git.example.com    # omit for public github.com
    export ANTHROPIC_AUTH_TOKEN=...       # required for chat
    export ANTHROPIC_BASE_URL=...         # optional, has a default
    export ANTHROPIC_MODEL=...            # optional, has a default
    cd GitDigger/kb_graph
    python workflow/server.py --port 8766
    open http://localhost:8766
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

KB_GRAPH_ROOT = Path(__file__).resolve().parent.parent
DISCOVERY_DIR = KB_GRAPH_ROOT / "discovery"

# kb_graph/ for ingest.py, query.py, common/*; discovery/ for harness.py,
# because harness.py itself does bare `from list_github_repos import ...` /
# `from relevance import ...` imports (it assumes discovery/ is already on
# sys.path, same as when discovery/server.py is run directly) rather than
# `from discovery.list_github_repos import ...`. Importing it as a bare
# `harness` module (not `discovery.harness`) keeps that assumption true here
# too, instead of half-fixing it and leaving it broken for the standalone
# discovery/server.py entry point.
sys.path.insert(0, str(KB_GRAPH_ROOT))
sys.path.insert(0, str(DISCOVERY_DIR))

import harness  # noqa: E402
import ingest  # noqa: E402
import query  # noqa: E402
from common.clone import CloneError, clone_repo  # noqa: E402
from common.context import build_context  # noqa: E402
from common.llm_client import ChatError, ask  # noqa: E402
from finetune.dataset_gen import DEFAULT_NUM_FACTS, generate_training_data  # noqa: E402
from finetune.errors import FinetuneError  # noqa: E402
from finetune.inference import generate_reply, get_or_load  # noqa: E402
from finetune.model_cache import is_model_cached  # noqa: E402
from finetune.runs import fail_run, finish_run, list_runs, start_run  # noqa: E402
from finetune.train import run_finetune  # noqa: E402

# In-memory background-job tracking for the fine-tune step, and a
# process-wide loaded-model cache inside finetune/inference.py itself — the
# one deliberate exception to this server's "stateless, browser holds
# state" convention used everywhere else in this file. A LoRA training run
# takes minutes and an 8B-parameter model can't be reloaded per request, so
# both need to outlive any single HTTP request.
_finetune_jobs: dict[str, dict] = {}
_finetune_jobs_lock = threading.Lock()

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a software "
    "system, grounded in a knowledge graph built from that system's "
    "Terraform infrastructure and Confluence design docs. You will be given "
    "a block of context retrieved from that graph before each question. "
    "Answer using only that context and the rest of this conversation. If "
    "the context doesn't contain enough information to answer, say so "
    "plainly instead of guessing or fabricating details."
)

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kb_graph workflow</title>
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
    --good: #2f8f4e;
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
      --good: #4fbf74;
      --bad: #e67567;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: var(--page); color: var(--text-primary);
    font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  #app { max-width: 860px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  h2 { font-size: 14px; margin: 0 0 8px; }
  .sub { color: var(--text-muted); font-size: 12px; margin-bottom: 20px; }
  section {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 16px; display: grid; gap: 10px;
  }
  section.disabled { opacity: .45; pointer-events: none; }
  .step-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); }
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
  .status { color: var(--text-secondary); font-size: 12px; min-height: 16px; }
  .status.error { color: var(--bad); }
  .status.ok { color: var(--good); }
  .card {
    border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px;
  }
  .card h3 { margin: 0 0 4px; font-size: 13px; display: flex; align-items: center; gap: 8px; }
  .card h3 a { color: var(--text-primary); text-decoration: none; }
  .card h3 a:hover { text-decoration: underline; }
  .badge {
    display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: .03em;
    color: var(--text-muted); border: 1px solid var(--border); border-radius: 3px;
    padding: 1px 5px; vertical-align: middle;
  }
  .badge.good { color: var(--good); border-color: var(--good); }
  .badge.bad { color: var(--bad); border-color: var(--bad); }
  .desc { color: var(--text-secondary); }
  .why { color: var(--text-muted); font-size: 11px; }
  #resultsList { display: grid; gap: 8px; }
  #cloneStatus, #ingestStatus { display: grid; gap: 6px; }
  #chatLog { display: grid; gap: 8px; max-height: 360px; overflow-y: auto; padding: 4px; }
  .msg { padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border); }
  .msg.user { background: var(--page); }
  .msg.assistant { background: var(--surface-1); }
  .msg b { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--text-muted); margin-bottom: 3px; }
  .msg-content { white-space: pre-wrap; word-break: break-word; }
  .msg-code {
    background: var(--page); border: 1px solid var(--border); border-radius: 5px;
    padding: 8px 10px; overflow-x: auto; margin: 6px 0;
    font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre;
  }
  .chatRow { display: grid; grid-template-columns: 1fr 100px; gap: 8px; }
</style>
</head>
<body>
<div id="app">
  <h1>kb_graph workflow</h1>
  <div class="sub">Discover related repos, clone the ones you want, build the graph, then ask it questions.</div>

  <section id="stepWorkspace">
    <div class="step-label">Step 0 — Workspace</div>
    <label for="workspace">Workspace directory (used for clones and the ingested graph)</label>
    <input id="workspace" placeholder="/path/to/a/scratch/directory" required>
  </section>

  <section id="stepDiscover">
    <div class="step-label">Step 1 — Discover</div>
    <label for="root_repo">Root repo (local path, already cloned)</label>
    <input id="root_repo" placeholder="/path/to/already-cloned/repo" required>
    <div style="display:flex; gap:16px; align-items:center;">
      <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;color:var(--text-secondary);font-size:13px;">
        <input type="radio" name="mode" value="topic" id="modeTopic" checked style="width:auto;">
        By topic
      </label>
      <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;color:var(--text-secondary);font-size:13px;">
        <input type="radio" name="mode" value="all" id="modeAll" style="width:auto;">
        All referenced repos
      </label>
    </div>
    <div id="topicField">
      <label for="topic">Topic</label>
      <input id="topic" placeholder="e.g. session stickiness">
    </div>
    <div class="row">
      <div>
        <label for="org">Org override (optional)</label>
        <input id="org" placeholder="leave blank to auto-detect">
      </div>
      <div>
        <label for="max_repos">Max repos</label>
        <input id="max_repos" type="number" min="1" value="10">
      </div>
    </div>
    <button id="discoverBtn" type="button">Discover</button>
    <div id="discoverStatus" class="status"></div>
    <div id="resultsList"></div>
  </section>

  <section id="stepClone" class="disabled">
    <div class="step-label">Step 2 — Clone</div>
    <button id="cloneBtn" type="button">Clone selected</button>
    <div id="cloneStatus" class="status"></div>
  </section>

  <section id="stepIngest" class="disabled">
    <div class="step-label">Step 3 — Ingest</div>
    <div class="sub" style="margin-bottom:0;">Builds the graph from the root repo, every cloned repo, and any extra local sources below.</div>
    <label for="extra_sources">Extra local sources (one path per line — repos "Discover" can't reach via module refs, design-doc folders, etc.)</label>
    <textarea id="extra_sources" rows="2" placeholder="/path/to/another/repo&#10;/path/to/a/design-doc/folder" style="width:100%; font: inherit; padding: 7px 9px; border-radius: 5px; border: 1px solid var(--border); background: var(--page); color: var(--text-primary);"></textarea>
    <button id="ingestBtn" type="button">Run ingest</button>
    <a id="viewGraphBtn" href="#" target="_blank" rel="noopener" style="display:none;">
      <button type="button">View graph &rarr;</button>
    </a>
    <div id="ingestStatus" class="status"></div>
  </section>

  <section id="stepChat" class="disabled">
    <div class="step-label">Step 4 — Chat (Claude, grounded)</div>
    <div id="chatLog"></div>
    <div class="chatRow">
      <input id="chatInput" placeholder="Ask a question about the ingested repos/docs...">
      <button id="chatBtn" type="button">Send</button>
    </div>
    <div id="chatStatus" class="status"></div>
  </section>

  <section id="stepGenData" class="disabled">
    <div class="step-label">Step 5 — Generate training data</div>
    <div class="sub" style="margin-bottom:0;">Asks Claude to propose design-rationale facts from the ingested graph, then expands each into paraphrased Q&amp;A pairs for fine-tuning.</div>
    <label for="num_facts">Number of facts</label>
    <input id="num_facts" type="number" min="1" value="3" style="max-width:140px;">
    <button id="genDataBtn" type="button">Generate</button>
    <div id="genDataStatus" class="status"></div>
    <div id="genDataLinks"></div>
  </section>

  <section id="stepFinetune" class="disabled">
    <div class="step-label">Step 6 — Fine-tune</div>
    <label for="ft_model_id">Base model</label>
    <input id="ft_model_id" value="meta-llama/Llama-3.1-8B-Instruct">
    <div id="modelCacheStatus" class="status"></div>
    <button id="startFinetuneBtn" type="button">Start fine-tuning</button>
    <div id="finetuneStatus" class="status"></div>
  </section>

  <section id="stepFinetuneChat">
    <div class="step-label">Step 7 — Chat with the fine-tuned model</div>
    <div class="sub" style="margin-bottom:0;">Pick any past run for this workspace below — this list doesn't require walking through the earlier steps again, and survives a server restart.</div>
    <label for="ftRunPicker">Fine-tuned run (persists across server restarts)</label>
    <div class="row" style="grid-template-columns: 1fr 100px;">
      <select id="ftRunPicker"><option value="">— no runs yet —</option></select>
      <button id="ftRunRefreshBtn" type="button">Refresh</button>
    </div>
    <div id="ftRunStatus" class="status"></div>
    <div id="ftChatLog"></div>
    <div class="chatRow">
      <input id="ftChatInput" placeholder="Ask one of the eval questions, without the excerpt...">
      <button id="ftChatBtn" type="button" disabled>Send</button>
    </div>
    <div id="ftChatStatus" class="status"></div>
  </section>
</div>
<script>
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const discoverBtn = $("discoverBtn"), cloneBtn = $("cloneBtn"), ingestBtn = $("ingestBtn"), chatBtn = $("chatBtn");
  const stepClone = $("stepClone"), stepIngest = $("stepIngest"), stepChat = $("stepChat");
  const stepGenData = $("stepGenData"), stepFinetune = $("stepFinetune");
  const genDataBtn = $("genDataBtn"), startFinetuneBtn = $("startFinetuneBtn"), ftChatBtn = $("ftChatBtn");
  const ftModelIdInput = $("ft_model_id");
  const ftRunPicker = $("ftRunPicker"), ftRunRefreshBtn = $("ftRunRefreshBtn");
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const topicField = $("topicField"), topicInput = $("topic");


  let discovered = [];       // last /api/discover "matched" list
  let clonedPaths = [];      // local paths of successfully cloned repos
  let graphOutDir = null;    // set once ingest succeeds
  let chatHistory = [];      // [{role, content}, ...]
  let trainDatasetPath = null;  // set once Step 5 generates data
  let ftJobId = null;            // set once Step 6 starts a fine-tune job
  let ftAdapterDir = null;       // adapter dir of the run selected in Step 7's picker
  let ftChatModelId = null;      // base model_id that run was trained from (independent of ft_model_id, which is for *starting* new runs)
  let ftPollTimer = null;
  let ftChatHistory = [];        // [{role, content}, ...] for Step 7's own panel
  let ftRunsById = {};            // job_id -> run record, from the last /api/finetune/runs fetch


  function applyModeVisibility() {
    const isAll = document.querySelector('input[name="mode"]:checked').value === "all";
    topicField.style.display = isAll ? "none" : "";
  }
  modeRadios.forEach(r => r.addEventListener("change", applyModeVisibility));
  applyModeVisibility();

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function setStatus(el, text, cls) {
    el.className = "status" + (cls ? " " + cls : "");
    el.textContent = text;
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("request failed (HTTP " + res.status + ")"));
    return data;
  }

  discoverBtn.addEventListener("click", async () => {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const payload = {
      root_repo: $("root_repo").value.trim(),
      mode,
      topic: mode === "topic" ? topicInput.value.trim() : null,
      org: $("org").value.trim() || null,
      max_repos: Number($("max_repos").value) || 10,
    };
    if (!payload.root_repo) { setStatus($("discoverStatus"), "root repo is required", "error"); return; }
    if (mode === "topic" && !payload.topic) { setStatus($("discoverStatus"), "topic is required in 'by topic' mode", "error"); return; }

    discoverBtn.disabled = true;
    setStatus($("discoverStatus"), "searching...");
    $("resultsList").innerHTML = "";
    try {
      const data = await postJSON("/api/discover", payload);
      discovered = data.matched;
      setStatus($("discoverStatus"),
        `org: ${data.org} · ${data.org_repo_count} repos in org · ${data.matched.length} candidate(s)`, "ok");
      if (!discovered.length) {
        $("resultsList").innerHTML = '<div class="card">No repos resolved. You can still proceed with just the root repo.</div>';
      } else {
        $("resultsList").innerHTML = discovered.map((m, i) => `
          <div class="card">
            <h3>
              <input type="checkbox" class="repoPick" data-i="${i}" style="width:auto;" checked>
              <a href="${esc(m.html_url)}" target="_blank" rel="noopener">${esc(m.repo)}</a>
              <span class="badge ${m.private ? "bad" : "good"}">${m.private ? "private" : "public"}</span>
            </h3>
            ${m.description ? `<div class="desc">${esc(m.description)}</div>` : ""}
          </div>
        `).join("");
      }
      stepClone.classList.remove("disabled");
    } catch (err) {
      setStatus($("discoverStatus"), err.message, "error");
    } finally {
      discoverBtn.disabled = false;
    }
  });

  cloneBtn.addEventListener("click", async () => {
    const workspace = $("workspace").value.trim();
    if (!workspace) { setStatus($("cloneStatus"), "set a workspace directory in Step 0 first", "error"); return; }
    const picked = Array.from(document.querySelectorAll(".repoPick:checked"))
      .map(el => discovered[Number(el.dataset.i)])
      .map(m => ({ full_name: m.repo, html_url: m.html_url }));

    cloneBtn.disabled = true;
    setStatus($("cloneStatus"), picked.length ? "cloning..." : "no repos selected — skipping to ingest with just the root repo");
    try {
      let results = [];
      if (picked.length) {
        const data = await postJSON("/api/clone", { repos: picked, dest_dir: workspace + "/clones" });
        results = data.results;
      }
      clonedPaths = results.filter(r => r.status !== "error").map(r => r.path);
      $("cloneStatus").innerHTML = "";
      const list = document.createElement("div");
      list.innerHTML = results.map(r => `<div class="why">${esc(r.repo)} — ${esc(r.status)}${r.error ? ": " + esc(r.error) : ""}</div>`).join("")
        || '<div class="why">nothing to clone</div>';
      $("cloneStatus").appendChild(list);
      stepIngest.classList.remove("disabled");
    } catch (err) {
      setStatus($("cloneStatus"), err.message, "error");
    } finally {
      cloneBtn.disabled = false;
    }
  });

  ingestBtn.addEventListener("click", async () => {
    const workspace = $("workspace").value.trim();
    const rootRepo = $("root_repo").value.trim();
    if (!workspace || !rootRepo) { setStatus($("ingestStatus"), "workspace and root repo are required", "error"); return; }

    ingestBtn.disabled = true;
    setStatus($("ingestStatus"), "ingesting...");
    try {
      const extraSources = $("extra_sources").value.split("\n").map(s => s.trim()).filter(Boolean);
      const sources = [rootRepo, ...clonedPaths, ...extraSources];
      const out = workspace + "/output";
      const data = await postJSON("/api/ingest", { sources, out });
      graphOutDir = data.out;
      setStatus($("ingestStatus"),
        `nodes: ${data.nodes} · edges: ${data.edges} · repos: ${data.repos.join(", ") || "-"} · wrote ${data.graph_json}`, "ok");
      const viewGraphBtn = $("viewGraphBtn");
      viewGraphBtn.href = "/graph-view?dir=" + encodeURIComponent(graphOutDir);
      viewGraphBtn.style.display = "";
      stepChat.classList.remove("disabled");
      stepGenData.classList.remove("disabled");
    } catch (err) {
      setStatus($("ingestStatus"), err.message, "error");
    } finally {
      ingestBtn.disabled = false;
    }
  });

  function formatMessage(text) {
    // Not a full markdown renderer (kb_graph avoids extra deps) — just
    // enough to keep chat replies legible: fenced ```code``` blocks render
    // monospace/pre, everything else keeps its line breaks via the
    // .msg-content { white-space: pre-wrap } CSS rule below.
    const parts = String(text).split(/```[^\n`]*\n?([\s\S]*?)```/g);
    // split() with one capturing group interleaves matches into the array:
    // [plainText0, code0, plainText1, code1, ..., plainTextN]
    return parts.map((part, i) =>
      i % 2 === 1 ? `<pre class="msg-code"><code>${esc(part)}</code></pre>` : esc(part)
    ).join("");
  }

  function renderChat() {
    $("chatLog").innerHTML = chatHistory.map(m => `
      <div class="msg ${m.role}"><b>${m.role}</b><div class="msg-content">${formatMessage(m.content)}</div></div>
    `).join("");
    $("chatLog").scrollTop = $("chatLog").scrollHeight;
  }

  async function sendChat() {
    const input = $("chatInput");
    const message = input.value.trim();
    if (!message) return;
    if (!graphOutDir) { setStatus($("chatStatus"), "run ingest first", "error"); return; }

    chatHistory.push({ role: "user", content: message });
    renderChat();
    input.value = "";
    chatBtn.disabled = true;
    setStatus($("chatStatus"), "thinking...");
    try {
      const data = await postJSON("/api/chat", {
        message,
        history: chatHistory.slice(0, -1),
        graph_out_dir: graphOutDir,
      });
      chatHistory.push({ role: "assistant", content: data.reply });
      renderChat();
      setStatus($("chatStatus"), "");
    } catch (err) {
      setStatus($("chatStatus"), err.message, "error");
    } finally {
      chatBtn.disabled = false;
    }
  }
  chatBtn.addEventListener("click", sendChat);
  $("chatInput").addEventListener("keydown", (ev) => { if (ev.key === "Enter") sendChat(); });

  // -- Step 5: generate training data --------------------------------

  genDataBtn.addEventListener("click", async () => {
    if (!graphOutDir) { setStatus($("genDataStatus"), "run ingest first", "error"); return; }
    const numFacts = Number($("num_facts").value) || 3;

    genDataBtn.disabled = true;
    setStatus($("genDataStatus"), "asking the model to propose facts, then expanding paraphrases (this can take a little while)...");
    $("genDataLinks").innerHTML = "";
    try {
      const data = await postJSON("/api/finetune/generate_data", { graph_out_dir: graphOutDir, num_facts: numFacts });
      trainDatasetPath = data.train_dataset;
      setStatus($("genDataStatus"),
        `facts: ${data.facts.join(", ")} · ${data.train_examples} train example(s) · ${data.eval_examples} eval question(s)`, "ok");
      const viewURL = (p) => "/api/finetune/view_file?path=" + encodeURIComponent(p) + "&workspace=" + encodeURIComponent(graphOutDir);
      $("genDataLinks").innerHTML = `
        <div class="why"><a href="${esc(viewURL(data.train_dataset))}" target="_blank" rel="noopener">${esc(data.train_dataset)}</a></div>
        <div class="why"><a href="${esc(viewURL(data.eval_questions))}" target="_blank" rel="noopener">${esc(data.eval_questions)}</a></div>
      `;
      stepFinetune.classList.remove("disabled");
      checkModelCache();
      loadFtRuns();
    } catch (err) {
      setStatus($("genDataStatus"), err.message, "error");
    } finally {
      genDataBtn.disabled = false;
    }
  });

  // -- Step 6: fine-tune ----------------------------------------------

  async function checkModelCache() {
    const modelId = ftModelIdInput.value.trim();
    if (!modelId) return;
    setStatus($("modelCacheStatus"), "checking local cache...");
    try {
      const data = await postJSON("/api/finetune/check_model", { model_id: modelId });
      setStatus($("modelCacheStatus"),
        data.cached ? `already cached at ${data.cache_dir}` : `not cached yet at ${data.cache_dir} — starting will download it (several GB)`,
        data.cached ? "ok" : "");
    } catch (err) {
      setStatus($("modelCacheStatus"), err.message, "error");
    }
  }
  ftModelIdInput.addEventListener("change", checkModelCache);

  startFinetuneBtn.addEventListener("click", async () => {
    const workspace = $("workspace").value.trim();
    if (!trainDatasetPath) { setStatus($("finetuneStatus"), "generate training data first", "error"); return; }
    const modelId = ftModelIdInput.value.trim();
    if (!modelId) { setStatus($("finetuneStatus"), "base model is required", "error"); return; }

    startFinetuneBtn.disabled = true;
    setStatus($("finetuneStatus"), "starting...");
    try {
      const runsDir = workspace + "/output/finetune/runs";
      const data = await postJSON("/api/finetune/start", {
        dataset_path: trainDatasetPath, model_id: modelId, runs_dir: runsDir,
      });
      ftJobId = data.job_id;
      if (ftPollTimer) clearInterval(ftPollTimer);
      ftPollTimer = setInterval(pollFinetuneStatus, 2000);
      pollFinetuneStatus();
    } catch (err) {
      setStatus($("finetuneStatus"), err.message, "error");
      startFinetuneBtn.disabled = false;
    }
  });

  async function pollFinetuneStatus() {
    if (!ftJobId) return;
    try {
      const res = await fetch("/api/finetune/status?job_id=" + encodeURIComponent(ftJobId));
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "status check failed");

      const latest = data.latest || {};
      const stepInfo = latest.step != null
        ? ` · step ${latest.step}${latest.total_steps ? "/" + latest.total_steps : ""}${latest.loss != null ? " · loss " + latest.loss.toFixed(4) : ""}`
        : "";
      setStatus($("finetuneStatus"), `[${data.status}] ${latest.message || latest.phase || ""}${stepInfo}`,
        data.status === "error" ? "error" : (data.status === "done" ? "ok" : ""));

      if (data.status === "done") {
        clearInterval(ftPollTimer);
        ftPollTimer = null;
        startFinetuneBtn.disabled = false;
        // Refresh the picker so the run that just finished shows up, and
        // select it automatically — the case someone will hit right after
        // "Start fine-tuning" completes, most of the time.
        loadFtRuns(ftJobId);
      } else if (data.status === "error") {
        clearInterval(ftPollTimer);
        ftPollTimer = null;
        startFinetuneBtn.disabled = false;
      }
    } catch (err) {
      setStatus($("finetuneStatus"), err.message, "error");
      clearInterval(ftPollTimer);
      ftPollTimer = null;
      startFinetuneBtn.disabled = false;
    }
  }

  // -- Step 7: chat with the fine-tuned model --------------------------

  async function loadFtRuns(selectJobId) {
    const workspace = $("workspace").value.trim();
    if (!workspace) return;
    const runsDir = workspace + "/output/finetune/runs";
    setStatus($("ftRunStatus"), "loading past runs...");
    try {
      const res = await fetch("/api/finetune/runs?runs_dir=" + encodeURIComponent(runsDir));
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "failed to list runs");

      ftRunsById = {};
      (data.runs || []).forEach(r => { ftRunsById[r.job_id] = r; });

      if (!data.runs || !data.runs.length) {
        ftRunPicker.innerHTML = '<option value="">— no runs yet —</option>';
        setStatus($("ftRunStatus"), "no fine-tuning runs recorded for this workspace yet");
        return;
      }

      ftRunPicker.innerHTML = data.runs.map(r => {
        const lossPart = r.final_loss != null ? ` · loss ${r.final_loss.toFixed(4)}` : "";
        const label = `${r.model_id} — ${r.created_at} — ${r.status}${lossPart}`;
        return `<option value="${esc(r.job_id)}" ${r.status !== "done" ? "disabled" : ""}>${esc(label)}</option>`;
      }).join("");

      // Prefer selecting: the job just passed in (a run that just
      // finished), else whatever was already selected if it's still
      // there, else the newest done run, else leave the placeholder.
      const preferred = selectJobId || ftRunPicker.value;
      const fallbackDone = data.runs.find(r => r.status === "done");
      const toSelect = (preferred && ftRunsById[preferred] && ftRunsById[preferred].status === "done")
        ? preferred
        : (fallbackDone ? fallbackDone.job_id : "");
      ftRunPicker.value = toSelect;
      applyFtRunSelection();
      setStatus($("ftRunStatus"), "");
    } catch (err) {
      setStatus($("ftRunStatus"), err.message, "error");
    }
  }

  function applyFtRunSelection() {
    const run = ftRunsById[ftRunPicker.value];
    if (run && run.status === "done") {
      ftAdapterDir = run.adapter_dir;
      ftChatModelId = run.model_id;
      ftChatBtn.disabled = false;
      setStatus($("ftChatStatus"), "");
    } else {
      ftAdapterDir = null;
      ftChatModelId = null;
      ftChatBtn.disabled = true;
    }
  }
  ftRunPicker.addEventListener("change", applyFtRunSelection);
  ftRunRefreshBtn.addEventListener("click", () => loadFtRuns());

  // The run picker itself needs no earlier step completed — it lists
  // whatever's already recorded for this workspace, restart or not — so
  // load it as soon as there's a workspace to look under, and again
  // whenever that field changes.
  loadFtRuns();
  $("workspace").addEventListener("change", () => loadFtRuns());

  function ftRenderChat() {
    $("ftChatLog").innerHTML = ftChatHistory.map(m => `
      <div class="msg ${m.role}"><b>${m.role}</b><div class="msg-content">${formatMessage(m.content)}</div></div>
    `).join("");
    $("ftChatLog").scrollTop = $("ftChatLog").scrollHeight;
  }

  async function sendFtChat() {
    const input = $("ftChatInput");
    const message = input.value.trim();
    if (!message) return;
    if (!ftAdapterDir || !ftChatModelId) { setStatus($("ftChatStatus"), "pick a fine-tuned run above first", "error"); return; }

    ftChatHistory.push({ role: "user", content: message });
    ftRenderChat();
    input.value = "";
    ftChatBtn.disabled = true;
    setStatus($("ftChatStatus"), "loading the fine-tuned model — the first message can take a minute...");
    try {
      const data = await postJSON("/api/finetune/chat", {
        message,
        history: ftChatHistory.slice(0, -1),
        model_id: ftChatModelId,
        adapter_dir: ftAdapterDir,
      });
      ftChatHistory.push({ role: "assistant", content: data.reply });
      ftRenderChat();
      setStatus($("ftChatStatus"), "");
    } catch (err) {
      setStatus($("ftChatStatus"), err.message, "error");
    } finally {
      ftChatBtn.disabled = false;
    }
  }
  ftChatBtn.addEventListener("click", sendFtChat);
  $("ftChatInput").addEventListener("keydown", (ev) => { if (ev.key === "Enter") sendFtChat(); });
})();

</script>
</body>
</html>
"""


class WorkflowError(Exception):
    """Raised for bad request payloads at the route layer — same role as
    the other *Error classes, caught once per handler and turned into a
    JSON error response."""


class WorkflowHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise WorkflowError(f"invalid JSON body: {e}") from e

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/graph-view":
            self._handle_graph_view(parse_qs(parsed.query))
        elif parsed.path == "/api/finetune/status":
            self._handle_finetune_status(parse_qs(parsed.query))
        elif parsed.path == "/api/finetune/view_file":
            self._handle_finetune_view_file(parse_qs(parsed.query))
        elif parsed.path == "/api/finetune/runs":
            self._handle_finetune_runs(parse_qs(parsed.query))
        else:
            self.send_error(404)

    def _handle_graph_view(self, query: dict) -> None:
        # Serves the graph.html that ingest.py already wrote into the
        # workspace's output dir, so "View graph" works as a plain HTTP
        # link (new tab) instead of a file:// link — the latter is served
        # from a different origin than this page and some browsers refuse
        # to navigate to it from a link click, depending on local
        # security settings. graph.html itself stays exactly what
        # ingest.py produced: self-contained, no server required if opened
        # directly by double-click either.
        dirs = query.get("dir")
        if not dirs or not dirs[0]:
            self.send_error(400, "missing 'dir' query parameter")
            return
        graph_html_path = Path(dirs[0]) / "graph.html"
        if not graph_html_path.is_file():
            self.send_error(404, f"no graph.html at {graph_html_path} — run ingest first")
            return
        body = graph_html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_finetune_status(self, query: dict) -> None:
        job_ids = query.get("job_id")
        if not job_ids or not job_ids[0]:
            self._send_json(400, {"error": "missing 'job_id' query parameter"})
            return
        with _finetune_jobs_lock:
            job = _finetune_jobs.get(job_ids[0])
        if job is None:
            self._send_json(404, {"error": f"no such job: {job_ids[0]!r}"})
            return
        self._send_json(200, job)

    def _handle_finetune_view_file(self, query: dict) -> None:
        # Serves a generated finetune/* file's raw text back to the browser
        # for Step 5's "view" links — same file://-origin problem
        # _handle_graph_view solves, but this route serves arbitrary file
        # *content* by caller-supplied path rather than one hardcoded
        # filename, so (unlike _handle_graph_view) it also enforces that
        # the resolved path is actually inside the workspace the browser
        # already told us about, not anywhere else on disk.
        paths = query.get("path")
        workspaces = query.get("workspace")
        if not paths or not paths[0]:
            self.send_error(400, "missing 'path' query parameter")
            return
        if not workspaces or not workspaces[0]:
            self.send_error(400, "missing 'workspace' query parameter")
            return
        try:
            resolved = Path(paths[0]).resolve()
            resolved.relative_to(Path(workspaces[0]).resolve())
        except ValueError:
            self.send_error(403, "path is outside the given workspace")
            return
        if not resolved.is_file():
            self.send_error(404, f"no such file: {resolved}")
            return
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_finetune_runs(self, query: dict) -> None:
        # Backs Step 7's model picker. Reads straight from the on-disk
        # manifest (finetune/runs.py) rather than the in-memory
        # _finetune_jobs dict, so past runs are still listed after a
        # server restart wiped that dict — the whole point of this route.
        runs_dirs = query.get("runs_dir")
        if not runs_dirs or not runs_dirs[0]:
            self._send_json(400, {"error": "missing 'runs_dir' query parameter"})
            return
        self._send_json(200, {"runs": list_runs(Path(runs_dirs[0]))})

    def do_POST(self) -> None:  # noqa: N802
        routes = {
            "/api/discover": self._handle_discover,
            "/api/clone": self._handle_clone,
            "/api/ingest": self._handle_ingest,
            "/api/chat": self._handle_chat,
            "/api/finetune/generate_data": self._handle_finetune_generate_data,
            "/api/finetune/check_model": self._handle_finetune_check_model,
            "/api/finetune/start": self._handle_finetune_start,
            "/api/finetune/chat": self._handle_finetune_chat,
        }
        handler = routes.get(self.path)
        if handler is None:
            self.send_error(404)
            return
        try:
            payload = self._read_json()
            result = handler(payload)
        except WorkflowError as e:
            self._send_json(400, {"error": str(e)})
        except (harness.DiscoveryError, ingest.IngestError, CloneError, ChatError, FinetuneError) as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:  # last-resort guard — surface, never crash the server
            self._send_json(500, {"error": f"unexpected error: {e}"})
        else:
            self._send_json(200, result)

    # -- route handlers -----------------------------------------------

    def _handle_discover(self, payload: dict) -> dict:
        root_repo = payload.get("root_repo")
        topic = payload.get("topic") or None
        org = payload.get("org") or None
        mode = payload.get("mode") or "topic"
        try:
            max_repos = int(payload.get("max_repos", 10))
        except (TypeError, ValueError):
            raise WorkflowError("max_repos must be a number")

        if mode not in ("topic", "all"):
            raise WorkflowError(f"invalid mode: {mode!r} — expected 'topic' or 'all'")
        if not root_repo:
            raise WorkflowError("root_repo is required")
        if mode == "topic" and not topic:
            raise WorkflowError("topic is required in 'topic' mode")

        return harness.run_discovery(Path(root_repo), topic, max_repos, org, mode=mode)

    def _handle_clone(self, payload: dict) -> dict:
        repos = payload.get("repos")
        dest_dir = payload.get("dest_dir")
        if not repos:
            raise WorkflowError("repos is required and must be non-empty")
        if not dest_dir:
            raise WorkflowError("dest_dir is required")

        # Same env vars discovery/ already requires — kb_graph's own GitHub
        # auth, never GitDigger's GITHUB_TOKEN/config.py default.
        pat = os.environ.get("GITHUB_PAT")
        if not pat:
            raise WorkflowError("GITHUB_PAT is not set")
        host = os.environ.get("GITHUB_HOST")

        dest = Path(dest_dir)
        results = []
        for repo in repos:
            full_name = repo.get("full_name")
            html_url = repo.get("html_url")
            if not full_name or not html_url:
                results.append({"repo": full_name or "?", "status": "error", "error": "missing full_name/html_url"})
                continue
            try:
                results.append(clone_repo(full_name, html_url, dest, pat, host))
            except CloneError as e:
                results.append({"repo": full_name, "status": "error", "error": str(e)})
        return {"results": results}

    def _handle_ingest(self, payload: dict) -> dict:
        sources = payload.get("sources")
        out = payload.get("out")
        if not sources:
            raise WorkflowError("sources is required and must be non-empty")
        if not out:
            raise WorkflowError("out is required")
        return ingest.run_ingest([Path(s) for s in sources], Path(out))

    def _handle_chat(self, payload: dict) -> dict:
        message = payload.get("message")
        history = payload.get("history") or []
        graph_out_dir = payload.get("graph_out_dir")
        if not message:
            raise WorkflowError("message is required")
        if not graph_out_dir:
            raise WorkflowError("graph_out_dir is required — run ingest first")

        graph_path = Path(graph_out_dir) / "graph.json"
        try:
            index = query.load_graph(graph_path)
        except query.QueryError as e:
            raise WorkflowError(str(e)) from e

        context = build_context(index, message)
        user_turn = f"Context retrieved from the knowledge graph:\n\n{context}\n\nQuestion: {message}"
        messages = [*history, {"role": "user", "content": user_turn}]
        reply = ask(CHAT_SYSTEM_PROMPT, messages)
        return {"reply": reply}

    def _handle_finetune_generate_data(self, payload: dict) -> dict:
        graph_out_dir = payload.get("graph_out_dir")
        try:
            num_facts = int(payload.get("num_facts", DEFAULT_NUM_FACTS))
        except (TypeError, ValueError):
            raise WorkflowError("num_facts must be a number")
        if not graph_out_dir:
            raise WorkflowError("graph_out_dir is required — run ingest first")

        graph_path = Path(graph_out_dir) / "graph.json"
        try:
            index = query.load_graph(graph_path)
        except query.QueryError as e:
            raise WorkflowError(str(e)) from e

        out_dir = Path(graph_out_dir) / "finetune"
        return generate_training_data(index, out_dir, num_facts=num_facts)

    def _handle_finetune_check_model(self, payload: dict) -> dict:
        model_id = payload.get("model_id")
        if not model_id:
            raise WorkflowError("model_id is required")
        return is_model_cached(model_id)

    def _handle_finetune_start(self, payload: dict) -> dict:
        dataset_path = payload.get("dataset_path")
        model_id = payload.get("model_id")
        runs_dir = payload.get("runs_dir")
        if not dataset_path:
            raise WorkflowError("dataset_path is required — generate training data first")
        if not model_id:
            raise WorkflowError("model_id is required")
        if not runs_dir:
            raise WorkflowError("runs_dir is required")

        # One directory per run (runs_dir/<job_id>) rather than a single
        # fixed adapter path — every past run's adapter stays on disk so
        # Step 7's picker (backed by finetune/runs.py's manifest, which
        # survives a server restart) has something to list.
        job_id = uuid.uuid4().hex
        runs_dir_path = Path(runs_dir)
        out_dir = start_run(runs_dir_path, job_id, model_id, dataset_path)

        with _finetune_jobs_lock:
            _finetune_jobs[job_id] = {"status": "running", "log": [], "latest": {}, "result": None, "error": None}

        def on_progress(update: dict) -> None:
            with _finetune_jobs_lock:
                job = _finetune_jobs.get(job_id)
                if job is None:
                    return
                job["log"].append(update)
                job["latest"] = update

        def run() -> None:
            try:
                result = run_finetune(Path(dataset_path), model_id, out_dir, on_progress=on_progress)
            except FinetuneError as e:
                with _finetune_jobs_lock:
                    _finetune_jobs[job_id]["status"] = "error"
                    _finetune_jobs[job_id]["error"] = str(e)
                fail_run(runs_dir_path, job_id, str(e))
            except Exception as e:  # noqa: BLE001 — a background thread has no caller to raise to
                with _finetune_jobs_lock:
                    _finetune_jobs[job_id]["status"] = "error"
                    _finetune_jobs[job_id]["error"] = f"unexpected error: {e}"
                fail_run(runs_dir_path, job_id, f"unexpected error: {e}")
            else:
                with _finetune_jobs_lock:
                    _finetune_jobs[job_id]["status"] = "done"
                    _finetune_jobs[job_id]["result"] = result
                finish_run(runs_dir_path, job_id, result)

        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job_id}

    def _handle_finetune_chat(self, payload: dict) -> dict:
        message = payload.get("message")
        history = payload.get("history") or []
        model_id = payload.get("model_id")
        adapter_dir = payload.get("adapter_dir")
        if not message:
            raise WorkflowError("message is required")
        if not model_id:
            raise WorkflowError("model_id is required")
        if not adapter_dir:
            raise WorkflowError("adapter_dir is required — finish fine-tuning first")

        model, tokenizer = get_or_load(model_id, adapter_dir)
        reply = generate_reply(model, tokenizer, message, history=history)
        return {"reply": reply}

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        print(f"[workflow] {self.address_string()} - {fmt % args}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    server = ThreadingHTTPServer(("localhost", args.port), WorkflowHandler)
    print(f"workflow UI: http://localhost:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
