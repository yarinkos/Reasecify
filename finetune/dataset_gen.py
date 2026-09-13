"""
Training-data generation for kb_graph's fine-tuning step.

Turns an already-ingested graph (repos + linked Confluence design docs) into
a LoRA fine-tuning dataset, the same shape proven out by the
llama-finetune-smoke-test project this feature is modeled on: a handful of
"design rationale" facts, each expressed as several paraphrased Terraform-
context questions (with a linked design-doc excerpt embedded, exactly like a
real workflow chat question would look), each repeated several times so the
total exposure per fact is correct as soon as one training epoch completes.

This is the one place in kb_graph/finetune/ that calls an LLM (via the
existing common/llm_client.ask — no new client, no new dependency) rather
than a local model: proposing *which* facts are interesting and how to
paraphrase them is exactly the kind of judgment call the rest of kb_graph
already delegates to Claude for chat, not something worth hand-rolling
scoring rules for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from common.llm_client import ask
from finetune.errors import FinetuneError

DEFAULT_NUM_FACTS = 3
DEFAULT_PARAPHRASES = 6
DEFAULT_REPEATS = 5

# Cap on how much raw node/doc text gets fed into the fact-proposal prompt —
# generous enough for a real repo+doc set, but bounded so a very large graph
# doesn't blow past the gateway's request size / context window.
MAX_NODE_LINES = 400
MAX_DOC_CHARS = 12000

SYSTEM_PROMPT = (
    "You are helping build a fine-tuning dataset that teaches a small "
    "language model the *design rationale* behind a piece of software "
    "infrastructure — not just what the Terraform declares, but why it's "
    "structured that way, per the linked design documentation. You will be "
    "given a summary of a knowledge graph (nodes from Terraform repos, plus "
    "the full text of any linked design docs). Propose distinct rationale "
    "facts a reader could only know by connecting a specific piece of "
    "infrastructure to a specific design-doc statement — not generic "
    "Terraform best-practice trivia. Respond with ONLY a JSON array, no "
    "prose before or after it, no markdown code fence."
)


def _build_graph_summary(index) -> str:
    """Render the whole graph as text for the fact-proposal prompt — every
    design_doc's full body (that's the source of the "why"), plus a
    name/type/source line for every other node (the source of the "what").
    Unlike common/context.py's build_context, there's no user question to
    score relevance against here, so this takes the graph as a whole rather
    than a top-K subset — capped (MAX_NODE_LINES / MAX_DOC_CHARS) so an
    unusually large graph still produces a bounded prompt.
    """
    lines = [
        f"Graph summary: {len(index.nodes)} nodes, "
        f"repos: {', '.join(index.meta.get('repos', [])) or '-'}, "
        f"{index.meta.get('confluence_pages', 0)} design doc(s).",
        "",
    ]

    doc_nodes = [n for n in index.nodes.values() if n.get("type") == "design_doc"]
    other_nodes = [n for n in index.nodes.values() if n.get("type") != "design_doc"]

    if doc_nodes:
        lines.append("=== Design doc(s) (full text) ===")
        for n in doc_nodes:
            text = n.get("attrs", {}).get("text", "")
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS] + "…[truncated]"
            lines.append(f"--- {n['label']} ---")
            lines.append(text)
            lines.append("")

    if other_nodes:
        lines.append("=== Infrastructure nodes ===")
        for n in other_nodes[:MAX_NODE_LINES]:
            src = n.get("source") or {}
            pointer = src.get("url") or src.get("path") or ""
            suffix = f" — {pointer}" if pointer else ""
            lines.append(f"- [{n['type']}] {n['label']} (id: {n['id']}){suffix}")
        if len(other_nodes) > MAX_NODE_LINES:
            lines.append(f"... and {len(other_nodes) - MAX_NODE_LINES} more node(s), omitted for length.")

    return "\n".join(lines)


def _build_user_prompt(graph_summary: str, num_facts: int, paraphrases: int) -> str:
    return f"""{graph_summary}

Propose exactly {num_facts} distinct design-rationale facts from the above.
For each fact, produce a JSON object with these exact keys:

- "fact": a short kebab_case identifier (e.g. "instance_group_manager_resilience").
- "expected_answer_gist": 2-4 sentences stating the rationale, grounded in
  both a specific graph node and a specific design-doc statement.
- "eval_question": one held-out question testing this fact — phrase it as
  "Given this Terraform context: ...\\n\\nExplain/Walk me through/..." the
  same way a real user would ask, but do NOT include the design-doc excerpt
  in this one question (it's used to test recall without the excerpt
  present).
- "paraphrases": an array of exactly {paraphrases} different phrasings of
  the same question, each structured as:
  "Given this Terraform context:\\n<node/resource/module reference lines>\\n"
  "Linked design doc excerpt: \\"<a real short quote from the design doc "
  "above>\\"\\n\\n<a question asking for the rationale>"
  — vary the wording/ordering across the {paraphrases} paraphrases, but keep
  each one a faithful, answerable question about the same fact.

Respond with a JSON array of exactly {num_facts} such objects. Nothing else.
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_facts_response(raw: str, num_facts: int, paraphrases: int) -> list[dict]:
    """Defensive JSON parsing — a model reply is never trusted to be clean
    JSON on the first try (a stray code fence or leading sentence is common
    even when explicitly told not to). Never lets a malformed reply become a
    stack trace; raises FinetuneError with the truncated raw reply instead,
    the same "never a stack trace" contract common/llm_client.ask already
    keeps one layer down.
    """
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        facts = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise FinetuneError(
            f"could not parse the LLM's fact proposal as JSON ({e}); "
            f"raw reply (truncated): {raw[:500]!r}"
        ) from e

    if not isinstance(facts, list) or not facts:
        raise FinetuneError(f"expected a non-empty JSON array of facts, got: {raw[:500]!r}")

    required_keys = {"fact", "expected_answer_gist", "eval_question", "paraphrases"}
    for i, item in enumerate(facts):
        if not isinstance(item, dict) or not required_keys.issubset(item):
            raise FinetuneError(
                f"fact #{i} is missing one of {sorted(required_keys)}: {item!r}"
            )
        if not isinstance(item["paraphrases"], list) or not item["paraphrases"]:
            raise FinetuneError(f"fact #{i} ('{item.get('fact')}') has no paraphrases")

    return facts


def generate_training_data(
    index,
    out_dir: Path,
    num_facts: int = DEFAULT_NUM_FACTS,
    paraphrases: int = DEFAULT_PARAPHRASES,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """Propose `num_facts` design-rationale facts from `index` (a
    query.GraphIndex already loaded from graph.json) via one LLM call, then
    expand them into two files under `out_dir`:

    - train_dataset.jsonl — num_facts * paraphrases * repeats lines, each
      {"messages": [{"role": "user", "content": <paraphrase>},
                     {"role": "assistant", "content": <expected_answer_gist>}]}
      — the exact shape finetune/train.py's load_examples() expects
      (json.loads(line)["messages"]), ported unchanged from the smoke test's
      train_lora.py.
    - eval_questions.jsonl — one line per fact,
      {"fact", "question": eval_question, "expected_answer_gist"} — the
      exact shape the smoke test's query_finetuned_no_excerpt.py already
      reads.

    Repeats are baked into train_dataset.jsonl itself (not left to
    num_train_epochs) so total exposure per fact is correct the moment one
    epoch finishes — the same guard against the "max_steps silently
    overrides num_train_epochs" HF footgun the smoke test called out.
    """
    if num_facts < 1:
        raise FinetuneError("num_facts must be at least 1")
    if paraphrases < 1:
        raise FinetuneError("paraphrases must be at least 1")
    if repeats < 1:
        raise FinetuneError("repeats must be at least 1")

    graph_summary = _build_graph_summary(index)
    user_prompt = _build_user_prompt(graph_summary, num_facts, paraphrases)
    raw = ask(SYSTEM_PROMPT, [{"role": "user", "content": user_prompt}], max_tokens=8192)
    facts = _parse_facts_response(raw, num_facts, paraphrases)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_dataset.jsonl"
    eval_path = out_dir / "eval_questions.jsonl"

    train_examples = 0
    with train_path.open("w") as f:
        for item in facts:
            answer = item["expected_answer_gist"]
            for paraphrase in item["paraphrases"]:
                example = {
                    "messages": [
                        {"role": "user", "content": paraphrase},
                        {"role": "assistant", "content": answer},
                    ]
                }
                for _ in range(repeats):
                    f.write(json.dumps(example) + "\n")
                    train_examples += 1

    with eval_path.open("w") as f:
        for item in facts:
            f.write(json.dumps({
                "fact": item["fact"],
                "question": item["eval_question"],
                "expected_answer_gist": item["expected_answer_gist"],
            }) + "\n")

    return {
        "facts": [item["fact"] for item in facts],
        "train_examples": train_examples,
        "eval_examples": len(facts),
        "train_dataset": str(train_path),
        "eval_questions": str(eval_path),
    }
