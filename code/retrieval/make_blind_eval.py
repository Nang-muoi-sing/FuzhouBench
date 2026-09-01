"""Build the blind naturalness-evaluation sheet for rebuttal Evidence 3.

Samples 50 LLM-generated colloquial queries (Qwen3-30B-A3B) and 50
human-written queries (manual_rewrites Group B), shuffles them with a
fixed seed, and writes:

  papers/data/blind_eval_sheet.md    annotator-facing; NO provenance
  papers/data/blind_eval_key.json    answer key; DO NOT open before rating

Annotation task (single native-speaker annotator, blind):
  For each query, rate 1-5 naturalness "how likely is this phrasing to be
  typed by a real user searching a Fuzhou dictionary?"
    5 = perfectly natural, exactly what a person would type
    4 = natural, minor awkwardness
    3 = acceptable but slightly stilted / bookish
    2 = awkward; understandable but no one would type this
    1 = clearly artificial or nonsensical

Usage:
  python papers/scripts/make_blind_eval.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SEED = 42
N_PER_GROUP = 50

llm_file = REPO / "rag-bot" / "data" / "colloquial_queries_qwen3_30b_a3b.json"
human_file = REPO / "rag-bot" / "data" / "eval_mandarin_dataset_v3.json"
out_sheet = REPO / "papers" / "data" / "blind_eval_sheet.md"
out_key = REPO / "papers" / "data" / "blind_eval_key.json"

rng = random.Random(SEED)

llm_raw = json.loads(llm_file.read_text(encoding="utf-8"))
llm_pool = [e["query"].strip() for e in llm_raw
            if e.get("query") and 2 <= len(e["query"].strip()) <= 30]
llm_sample = rng.sample(llm_pool, N_PER_GROUP)

human_raw = json.loads(human_file.read_text(encoding="utf-8"))
human_pool = [e["query"].strip() for e in human_raw
              if e.get("group") == "B" and len(e["query"].strip()) >= 2]
human_sample = rng.sample(human_pool, N_PER_GROUP)

items = [{"query": q, "source": "llm"} for q in llm_sample] + \
        [{"query": q, "source": "human"} for q in human_sample]
rng.shuffle(items)

# --- annotator sheet (no provenance) ---
lines = [
    "# Blind naturalness evaluation — 100 dictionary-search queries",
    "",
    "For each query, imagine a real person searching a Fuzhou-dialect",
    "dictionary. Rate how natural the phrasing is as something a real user",
    "would type, on a 1-5 scale:",
    "",
    "- **5** perfectly natural, exactly what a person would type",
    "- **4** natural, minor awkwardness",
    "- **3** acceptable but slightly stilted or bookish",
    "- **2** awkward; understandable but no one would type this",
    "- **1** clearly artificial or nonsensical",
    "",
    "Fill in the Score column. Do not consult the answer key until done.",
    "",
    "| # | Query | Score (1-5) |",
    "|---|---|---|",
]
for i, item in enumerate(items, 1):
    lines.append(f"| {i:03d} | {item['query']} |  |")
out_sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- answer key ---
key = {f"{i:03d}": item["source"] for i, item in enumerate(items, 1)}
out_key.write_text(json.dumps(
    {"seed": SEED, "n_per_group": N_PER_GROUP, "key": key},
    ensure_ascii=False, indent=2), encoding="utf-8")

n_llm = sum(1 for v in key.values() if v == "llm")
print(f"sheet: {out_sheet}  ({len(items)} items, {n_llm} llm / {len(items)-n_llm} human)")
print(f"key:   {out_key}  (do not open before rating)")
