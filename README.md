# Reasecify: The Context-Aware DevSec Companion

Reasecify (Reasoning + Security + Graph) connects all of an organization's
available data — infrastructure-as-code, design docs, and the decisions
behind them — into one place. By combining Graph-Based Infrastructure
Ingestion, Dynamic Knowledge Base (Wiki) Linking, and Domain-Specific LoRA
Fine-Tuning, Reasecify creates a dedicated AI companion that internalizes
your organization's unique architecture and security decisions.

```
┌───────────────────────────┐
│         Discover          │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│           Clone           │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│          Ingest           │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│  Knowledge Graph + Wiki   │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│  Chat (Claude, grounded)  │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│  Generate training data   │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│     Fine-tune (LoRA)      │
└───────────────────────────┘
             │
             ▼
┌───────────────────────────┐
│  Chat (fine-tuned model)  │
└───────────────────────────┘
```

- **Infrastructure Knowledge Graph** — Parses IaC (Terraform) to map
  dependencies, cross-repo references, and resource connections into a
  unified typed graph.

- **Unified Knowledge Base (Wiki)** — Ingests Confluence design docs
  alongside code, linking architecture decisions directly to live
  infrastructure nodes without siloed context.

- **Fine-Tuned Domain Intelligence** — Generates synthetic training
  datasets straight from your architecture graph to LoRA fine-tune local
  open-weight models (Llama 3), turning your team's design rationale into
  model weights.

- **Dual-Engine Chat** — Features grounded retrieval-augmented generation
  (RAG) for instant, context-verified queries, alongside fine-tuned local
  model inference for native architecture recall.

## How does it help me?

Each backbone on its own falls short:

- **A plain RAG setup** retrieves out of context in many cases — a single
  pass over unstructured text misses the relationships that live in the
  *structure* of the infrastructure itself, not just its words.
- **A plain wiki-LLM setup** doesn't scale — it needs all the relevant data
  handed to it fresh for every single question, with nothing carried over
  or connected between them.
- **Fine-tuning alone on raw data** can mislead — a model trained straight
  on undigested text can pick up noise and stale detail as if it were fact.

Reasecify combines all three backbones into one streamlined process, so you
get the best of each: real answers grounded in your organization's own
facts, a working understanding of your security architecture, and a head
start on finding and fixing the next issue.

## The workflow, end to end

Reasecify is a guided, one-page pipeline (`workflow/server.py`) that walks
through all of this in order:

1. **Discover** — starting from one repo you know, find the other repos in
   its org that are actually relevant to a topic, so you're not manually
   hunting for what to pull in.
2. **Clone** — pull the relevant repos down locally.
3. **Ingest** — parse every cloned repo's infrastructure-as-code *and* any
   linked design docs into one typed knowledge graph — the "wiki" and the
   "graph" becoming one connected structure, not two separate silos.
4. **Chat (Claude, grounded)** — ask questions against that graph
   immediately, with real answers grounded in retrieval, before any
   fine-tuning happens — a sanity check that the ingested knowledge is
   actually useful.
5. **Generate training data** — turn the graph's design rationale into a
   real training set: an LLM proposes domain-specific "facts" worth
   teaching, then expands each into paraphrased training examples.
6. **Fine-tune** — LoRA fine-tune a local model on that training set, so
   the organization's own reasoning becomes part of the model's weights,
   not just something it can look up.
7. **Chat with the fine-tuned model** — talk to the result directly, with
   no retrieval or grounding involved, to see whether the fine-tuned model
   actually recalls the trained facts on its own.

Steps 1–4 already work standalone as a "know what exists and why" tool.
Steps 5–7 are what turn that knowledge into a model that has internalized
it — the fine-tuning half of Reasecify's name.

---

This is a **first-pass prototype**: the ingest/discovery/query pieces have
no LLM calls at all (pure parsing and graph traversal); the workflow
module above is what layers chat, training-data generation, and
fine-tuning on top of that same graph.

## What it does

Point `ingest.py` at a folder. It walks the folder recursively and, by file
shape (not by fixed subfolder names):

- Parses any `.tf` file it finds with a small hand-rolled scanner (see
  `parsers/terraform.py`) — no `python-hcl2` dependency. It pulls out
  `resource`/`module`/`variable`/`output` block headers and attribute
  references (`type.name.attr`, `module.name.attr`, `var.name`). This is a
  line-oriented scanner, not a full HCL2 grammar parser — it's good enough to
  build a graph of "what refers to what," but it doesn't evaluate expressions
  or understand `dynamic`/`for_each` semantics.
- Parses any `.json` file that looks like a fetched Confluence page (has a
  `body.storage` or `body.view` key) — see `parsers/confluence.py`. Strips the
  HTML and keeps the plain text as a `design_doc` node.
- Links Terraform repos to each other via `module` block `source` strings
  (full significant-word-subset match against known repo directory names,
  after dropping generic vendor/cloud words like "tf"/"google"/"module" — a
  registry source string usually doesn't literally contain the repo's
  directory name, but shares its distinctive words; matching requires *every*
  one of a repo's significant words to appear in the source, not just some
  fixed-count overlap, to avoid confusing two differently-named repos that
  happen to share a few words). When no cloned repo's words are fully
  contained in a module's source, the call links to a synthesized
  `external_module` placeholder node instead of guessing or silently
  dropping the edge — the dependency is real even when its target isn't in
  this ingest's source folder.
- Treats each directory under a Terraform repo as its own root
  module/workspace (matching how `dev/`, `stg/`, `prd/` environment folders
  are typically structured) — so identically-named blocks declared in
  different environments become distinct nodes (labeled `[dev]`/`[stg]`/
  `[prd]`, etc.) instead of collapsing into one, and a `var.x`/`module.x`
  reference in one environment's files never resolves against another
  environment's declaration of the same name.
- Links design docs to Terraform nodes via keyword overlap (2+ shared
  keywords, or one shared keyword of 8+ characters, between the doc's text
  and a node's name/label) — no embeddings/LLM in v1.

Output goes to `output/` (gitignored — regenerate anytime):

- `graph.json` — the full node/edge graph. This is the schema/storage format
  other tooling (`query.py`, and a future `lint` step) builds on.
- `GRAPH_SUMMARY.md` — quick human-readable counts, for sanity-checking a run
  without opening the viewer.
- `graph.html` — a single self-contained viewer. Vanilla JS + `<canvas>`, a
  small hand-rolled force-directed layout, **no CDN scripts, no build step**.
  The graph data is inlined directly into the HTML at generation time, so
  you can just double-click the file open — no server, no CORS issues. Each
  of the 8 node types gets both a distinct marker shape and a color
  (validated with the dataviz skill's palette checker) so identity never
  depends on color alone; a legend, hover tooltips, and a table-view
  fallback are all part of the same reason.

## Usage

```bash
python ingest.py --source /path/to/folder --out output
open output/graph.html
```

## Querying the graph

`query.py` is the second operation in the ingest/query/lint model — it
reads `graph.json` and answers real questions instead of requiring you to pan
around the canvas viewer. `--graph` defaults to `output/graph.json` and works
either before or after the subcommand.

Every command takes a loose node reference — an id, an exact name, or a
substring — never the full `type:path/to/thing` id. An ambiguous or
unresolvable reference is a hard error (non-zero exit), never a silent guess.

```bash
# "I don't know the exact name" — substring search over name/label
python query.py find web_service --graph output/graph.json

# Full detail for one node: type, source, attrs, edge counts by type
python query.py show web_service_001 --graph output/graph.json

# What's connected to this node (1 hop out, by default)
python query.py neighbors web_service_001 --graph output/graph.json

# What a design doc covers / what covers a piece of infra — both directions
# of `documents` edges
python query.py documents "stickiness" --graph output/graph.json

# Shortest connection between two nodes, any edge type, either direction
python query.py path web_service_001 "tf-google-foundations-network-gateway-compute-module" --graph output/graph.json
```

Every subcommand also accepts `--json` for machine-readable output — the seam
for a future free-form natural-language layer (pipe a resolved subgraph to an
LLM), not built yet.

## Discovery module (experimental)

`ingest.py`/`query.py` assume you already know which repos to point at.
`discovery/` answers a different question: given one repo you already have
locally and a topic, which *other* repos in its GitHub org are worth pulling
in? It chases the root repo's Terraform `module` `source` strings — the same
signal `ingest.py`'s `calls_module` linking uses — ranks them by relevance to
the topic, and resolves the relevant ones against the org's actual repo
list.

```bash
export GITHUB_PAT=...           # required — a token with read access to the org
export GITHUB_HOST=git.example.com    # omit for public github.com
python discovery/server.py --port 8765
open http://localhost:8765
```

Fill in a local path to an already-cloned repo, how many repos to return, and
pick one of two modes:

- **By topic** (default) — also fill in a topic (free text — "proxy farm
  stickiness"). It:
  1. lists every repo in the org (org auto-detected from the root repo's
     `.git/config` origin remote, or pass `org` explicitly in the form),
  2. reads the root repo's Terraform `module` blocks,
  3. scores each module against the topic by shared-word overlap — a module
     with nothing in common with the topic (a generic `auth`/`logging`
     module, say) scores zero and drops out on its own; no hardcoded
     "generic module" list needed,
  4. resolves the surviving modules' `source` strings against the org's repo
     list (same significant-word-subset rule as `calls_module`, just matched
     against the org's live repo list instead of already-cloned directory
     names),
  5. returns the top N, each with the module that pointed at it and the
     words that made it relevant — never just a bare score.
  A "Show all scanned modules (debug)" checkbox reveals every module_call
  found in the root repo, including the zero-score ones that never made it
  past step 3 — so "why didn't module X show up" always has a visible
  answer instead of a silent gap.
- **All referenced repos** — no topic needed. Skips step 3 entirely: every
  module_call in the root repo is resolved against the org's repo list,
  regardless of relevance to anything. Useful for "what does this repo
  reference in this org, period" rather than chasing one topic. In this
  mode the debug checkbox is redundant — nothing was filtered out before
  resolution, so the main results already show everything.

You can also hit `POST /api/discover` directly (`{"root_repo", "mode",
"topic", "max_repos", "org"}` → JSON) without the browser UI — `topic` is
required when `mode` is `"topic"` (the default) and ignored/optional when
`mode` is `"all"`.

**Deliberately out of scope for this pass:** discovered repos are only
listed, never cloned; there's no graph/KB generation from them yet (a doc
mentions using this as an input to a per-topic `ingest.py` run later, but
that wiring doesn't exist); it's Terraform-only (same limitation as
`ingest.py`'s parsing); and it's scoped to one GitHub org, not a
cross-org/dependency-transitive crawl. (The `workflow/` module below wires
this into clone+ingest+chat — `discovery/` itself is unchanged and works
standalone exactly as described above.)

## `common/` — shared helpers

Code used by more than one of the modules above (or by `workflow/`) lives in
`common/`, not duplicated per-module:

- `common/net.py` — `build_ssl_context(insecure_env_var)`, the same
  verified-by-default/opt-out-by-env-var TLS policy `discovery/` already
  used, factored out so `common/llm_client.py` shares it instead of
  copy-pasting it.
- `common/llm_client.py` — a stdlib (`urllib`) Anthropic Messages API
  client: `ask(system_prompt, messages, max_tokens=2048) -> str`. Hand-rolled
  rather than adding the `anthropic` SDK as a dependency, to keep this
  project stdlib-only outside of `finetune/` (same spirit as
  `discovery/list_github_repos.py` hand-rolling a GitHub client instead of
  adding a GitHub SDK dependency). Reads `ANTHROPIC_AUTH_TOKEN` /
  `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` directly from the environment —
  point these at whatever gateway or account you use via your own `.env`,
  they're never hardcoded here. TLS opt-out (rare, internal-host-only) via
  `ANTHROPIC_INSECURE_TLS=1`.
- `common/clone.py` — `clone_repo(full_name, html_url, dest_dir, pat, host)`,
  a small `git clone` wrapper via `subprocess` (no GitPython), authenticating
  against a GitHub Enterprise host via this project's own
  `GITHUB_PAT`/`GITHUB_HOST` env vars. Idempotent — skips with `status:
  "already_present"` if the destination already has a `.git` directory.
- `common/context.py` — `build_context(index, user_message, top_k=15) ->
  str`, the retrieval step for chat. Scores every graph node against the
  latest message using `discovery/relevance.py`'s existing tokenize/overlap
  scoring (no new scoring rule, no embeddings), and renders the top matches
  as text — including a snippet read straight off disk around a node's
  `source.line` when that file is still present locally (cheap, since a
  repo is already cloned locally by the time chat is reachable).

## `workflow/` — Discover → Clone → Ingest → Chat → Fine-tune, one page

`discovery/server.py` only ever answers "which repos are relevant" and
`ingest.py`/`query.py` still assume you drive cloning and ingestion by hand.
`workflow/` is a **separate module** that orchestrates all of the above
(plus `common/` and `finetune/`) into one guided page — it imports
`discovery/harness.py`, `ingest.py`, `query.py`, `common/`, and `finetune/`
directly as libraries; it does not modify any of them, and
`discovery/server.py` keeps working standalone, unchanged, on its own port.

```bash
export GITHUB_PAT=...                 # required — same var discovery/ uses
export GITHUB_HOST=git.example.com    # omit for public github.com
export ANTHROPIC_AUTH_TOKEN=...       # required for the chat step and Generate training data
export ANTHROPIC_BASE_URL=...         # optional, has a default
export ANTHROPIC_MODEL=...            # optional, has a default
pip install -r requirements.txt       # only needed for Fine-tune/Chat-with-fine-tuned-model
python workflow/server.py --port 8766
open http://localhost:8766
```

The page walks through seven steps, gated in order:

1. **Discover** — same `run_discovery` call `discovery/server.py` uses
   (`by topic` or `all referenced repos` mode), against a root repo and a
   workspace directory you provide.
2. **Clone** — check which discovered repos to keep, then "Clone selected"
   clones each into `<workspace>/clones/`. Authenticates with this
   project's own `GITHUB_PAT`/`GITHUB_HOST`. Re-clicking is safe: an
   already-cloned repo is skipped, not re-cloned.
3. **Ingest** — "Run ingest" builds the graph from the root repo *plus*
   every successfully cloned repo (`ingest.run_ingest` takes a list of
   source roots, not just one) into `<workspace>/output/graph.json`. This
   is the step where the "wiki" (linked design docs) and the "graph"
   (infra-as-code) actually merge into one connected structure.
4. **Chat (Claude, grounded)** — once ingest succeeds, ask questions in a
   chat box. Each question is answered by retrieving relevant graph nodes
   (`common/context.py`) and sending them as grounding context to Claude
   (`common/llm_client.py`) alongside the conversation so far. The server
   holds no session state for this step — the browser resends the growing
   `history` each turn, matching every other server in this project's
   "server does one request, browser holds the state" convention.
5. **Generate training data** — asks Claude to propose a handful of
   design-rationale "facts" from the ingested graph (design docs + infra
   nodes), then expands each into several paraphrased Q&A pairs, repeated
   several times, and writes `train_dataset.jsonl` + `eval_questions.jsonl`
   under `<workspace>/output/finetune/` — see `finetune/` below. Both files
   get a clickable link once generated.
6. **Fine-tune** — LoRA fine-tune a local base model (default
   `meta-llama/Llama-3.1-8B-Instruct`) on `train_dataset.jsonl`. Shows
   whether the base model is already in the local Hugging Face cache before
   you start (`finetune/model_cache.py`), then runs training in a
   background thread while the page polls `/api/finetune/status` for live
   progress (phase/step/loss).
7. **Chat with the fine-tuned model** — a second chat panel, stacked
   directly below Step 4's, that talks to a fine-tuned adapter instead of
   Claude — no retrieval/grounding involved, since the point is testing
   whether the fine-tuned *weights* recall the trained facts. A picker
   above the chat log lists every fine-tuning run recorded for the current
   workspace (`finetune/runs.py`'s on-disk manifest under
   `<workspace>/output/finetune/runs/`), including runs from previous
   server processes — not just whatever the current browser session just
   trained — so you can come back later and pick an older adapter without
   re-running Step 6.

`POST /api/discover`, `/api/clone`, `/api/ingest`, `/api/chat`,
`/api/finetune/generate_data`, `/api/finetune/check_model`,
`/api/finetune/start`, and `/api/finetune/chat` (plus `GET
/api/finetune/status`, `/api/finetune/view_file`, and `/api/finetune/runs`)
on this server can also be hit directly with `curl` — each returns a clean
JSON `{"error": "..."}` on a bad request rather than a stack trace.

**Known v1 limitation**: chat retrieval is per-turn and latest-message-only
— each new question re-scores the graph from just that message's words, not
the whole conversation. A follow-up like "tell me more about that" won't
automatically re-pull the prior turn's matched nodes; it relies on the LLM's
own memory of what it already said, not on retrieval seeing the earlier
question's context again.

## `finetune/` — training-data generation, local LoRA fine-tuning, inference

This is where the "combine the wiki and the graph to fine-tune" idea
actually turns into a model: it's the one part of this project that isn't
stdlib-only (see `requirements.txt`: `torch`, `transformers`, `peft`,
`accelerate`, `huggingface_hub`), and the one place `workflow/server.py`
keeps real server-side state (a background job-status dict for the
fine-tune step, and a process-wide loaded-model cache) instead of the
"browser holds all the state" convention used everywhere else — an
8B-parameter model and a multi-minute training run can't be pushed through
a "browser resends everything" request/response cycle.

Design and hyperparameters are ported unchanged from a standalone
proof-of-concept that validated the approach end to end on a real ingested
graph plus a linked Confluence design doc: LoRA (`r=16`, `alpha=32`,
`dropout=0.05`, all attention + MLP projections), `num_train_epochs=1` with
repeat-exposure baked into the dataset itself instead (raising epochs would
multiply that exposure again — see `finetune/train.py`'s module
docstring), greedy decoding at inference time for reproducible eval.

- `finetune/errors.py` — `FinetuneError`, same role as `DiscoveryError` /
  `IngestError` / `CloneError` / `ChatError`.
- `finetune/dataset_gen.py` — `generate_training_data(index, out_dir,
  num_facts=3, paraphrases=6, repeats=5)`. The one call in this package that
  uses an LLM (`common/llm_client.ask`, not a local model) — proposing
  *which* facts are worth teaching and how to paraphrase them is a judgment
  call, not something worth hand-rolling scoring rules for.
- `finetune/model_cache.py` — `is_model_cached(model_id)`, a stdlib-only
  (no `huggingface_hub` import) check of whether a model is already in the
  local Hugging Face hub cache, so the UI can warn before kicking off a
  multi-GB download.
- `finetune/train.py` — `run_finetune(dataset_path, model_id, out_dir,
  on_progress)`, LoRA training with phase/step/loss progress callbacks.
  `workflow/server.py` passes a per-run `out_dir` (`<runs_dir>/<job_id>`,
  see `finetune/runs.py`) rather than one fixed path, so every run's
  adapter survives later runs instead of being overwritten by them.
- `finetune/inference.py` — `get_or_load(model_id, adapter_dir)` (loads +
  merges the adapter once per process, then caches it) and
  `generate_reply(model, tokenizer, message, history)` (greedy decode, no
  retrieval/grounding).
- `finetune/runs.py` — stdlib-only durable record of past fine-tuning runs:
  a JSON manifest at `<workspace>/output/finetune/runs/manifest.json`,
  appended to when a run starts and updated in place when it finishes or
  fails. Backs Step 7's run picker and the `GET /api/finetune/runs` route —
  the one place in this feature that needs to survive a `workflow/server.py`
  restart, since `_finetune_jobs` (the noisy per-step training log a
  *running* job's poll loop reads) stays in-memory only.

Generated artifacts (datasets, eval questions, adapters, training logs, the
runs manifest) are gitignored under `finetune/` — regenerate anytime;
nothing under `<workspace>/output/finetune/` should be committed.

## Known limitations (v1)

- Terraform parsing is regex/brace-based, not a real HCL parser — it can be
  fooled by unusual formatting (e.g. a `resource` keyword appearing inside a
  string or comment). Good enough for "what talks to what," not for anything
  that needs to be semantically exact.
- Each directory under a repo is treated as its own flat root module — a
  reasonable approximation of real `dev`/`stg`/`prd`-style layouts, but it
  will over-split a repo that genuinely shares declarations across
  directories via `terraform_remote_state` or symlinks (not modeled).
- `calls_module` matching requires every one of a repo's significant words to
  appear in the module's source string — deliberately strict to avoid
  cross-linking two differently-named repos, at the cost of missing a real
  match if a repo's directory name and its registry path diverge more than
  that (e.g. a repo renamed after publishing). A source with no confident
  match becomes an `external_module` placeholder node rather than a silent
  drop or a guess.
- Doc-to-infra linking is keyword overlap on node names (2+ shared keywords,
  or one shared keyword of 8+ characters) — no ranking, no disambiguation
  between similarly-named things across repos.
- No incremental/re-ingest logic yet — every run rebuilds the graph from
  scratch. Staleness/drift between a doc and the infra it describes is a
  real failure mode this doesn't detect yet; that's deliberately out of
  scope until this first pass is validated.
- `query.py` covers structured traversal only — no free-form natural-language
  questions yet (the `--json` flag is the seam for that later) and no `lint`
  / drift detection (e.g. doc-mentions-deleted-infra, infra-with-no-doc).
- `discovery/` inherits `calls_module`'s strict word-subset matching (see
  above) when resolving a module `source` against the org's repo list, so
  the same tradeoff applies: a real match is missed if a repo's directory
  name and its registry path diverge more than expected, rather than risking
  a wrong guess. It's also single-org and Terraform-only. Used on its own
  (`discovery/server.py`) it still stops at "here's the ranked list" — no
  cloning or KB generation; `workflow/server.py` is what takes a discovered
  list the rest of the way to clone+ingest+chat.
- `workflow/`'s chat is retrieval-by-keyword-overlap only (see
  `common/context.py` above) — no embeddings, no re-ranking across turns —
  and it never fetches file content from GitHub for chat context, only from
  a repo already cloned to local disk in the same session. Nothing in
  `workflow/` modifies `discovery/server.py`, which remains fully usable on
  its own.
- `finetune/`'s dataset generation asks Claude to propose facts and
  paraphrases in one call with no held-out review step — a bad or
  hallucinated proposal trains straight into the adapter. There's also no
  UI for actually running `eval_questions.jsonl` (it's generated for parity
  with the proof-of-concept this was ported from, but nothing in the
  workflow page runs it), and fine-tuning itself is single-GPU/MPS/CPU, one
  job at a time — `_finetune_jobs` has no queue or concurrency limit beyond
  "one background thread per Start click."
