"""
G2P baseline runner for OpenAI models via the Batch API (50% cheaper than
synchronous completions). Same eval-set construction and scoring as
g2p_baseline.py and g2p_baseline_remote.py, so results are directly
comparable to the Qwen3 numbers.

Workflow:
  submit  -> build JSONL, upload, create batch job, print batch ID
  poll    -> check status of an existing batch
  collect -> download finished batch output, score, write JSON

Run:
  python g2p_baseline_openai_batch.py submit \
      --model gpt-5.5 --n 500 --api-key-file <path>
  python g2p_baseline_openai_batch.py poll    --batch-id <id> --api-key-file <path>
  python g2p_baseline_openai_batch.py collect --batch-id <id> --model gpt-5.5 \
      --api-key-file <path>  --out papers/data/g2p_gpt5.5_batch.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd


FEWSHOT_EXAMPLES = [
    ("牛乳",   "ngu33 neing53"),      # 'cow-milk' = milk; primary reading (sandhi)
    ("講",     "goung33"),             # 'to speak'; single-syl primary
    ("白菜",   "bah21 jai213"),        # 'Chinese cabbage'; primary reading (sandhi, c->j)
    ("做什乇", "zo21 sie53 nooh24"), # 'do-what-thing' = to do what; primary (native speaker)
    ("我",     "nguai33"),             # 'I / me'; single-syl primary
]


def build_user_prompt(word: str) -> str:
    ex = "\n".join(f"Word: {w}\nYngping: {y}" for w, y in FEWSHOT_EXAMPLES)
    return f"Examples:\n{ex}\n\nWord: {word}\nYngping:"


SYSTEM_PROMPT = (
    "You convert Fuzhou Eastern Min words written in Chinese characters "
    "to their Yngping romanization. Each syllable in Yngping is letters "
    "followed by tone digits (one of: 55, 53, 33, 21, 24, 242, 213, 5, 0). "
    "Multi-syllable outputs are space-separated. "
    "Output ONLY the Yngping string, no commentary, no extra words."
)


YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")


def parse_yngping(s: str) -> list[tuple[str, str]]:
    s = s.strip().lower().split("\n")[0]
    syls = []
    for tok in s.split():
        m = YNGPING_TOKEN_RE.match(tok)
        if not m:
            continue
        syls.append((m.group(1), m.group(2)))
    return syls


def score_pair(pred: str, gold: str) -> dict:
    p = parse_yngping(pred)
    g = parse_yngping(gold)
    n = max(len(p), len(g))
    syl_correct = 0
    tone_correct = 0
    for i in range(n):
        ps = p[i] if i < len(p) else ("", "")
        gs = g[i] if i < len(g) else ("", "")
        if ps == gs:
            syl_correct += 1
        if ps[1] == gs[1] and gs[1] != "":
            tone_correct += 1
    return {"n_syl_gold": len(g), "n_syl_correct": syl_correct, "n_tone_correct": tone_correct}


def build_eval_set(resources_dir: Path, n: int, seed: int) -> list[dict]:
    pron = pd.read_csv(
        resources_dir / "PronunciationResource.tsv", sep="\t", dtype=str,
        keep_default_na=False, na_values=[""], encoding="utf-8", engine="python",
    )
    word = pd.read_csv(
        resources_dir / "WordResource.tsv", sep="\t", dtype=str,
        keep_default_na=False, na_values=[""], encoding="utf-8", engine="python",
    )
    primary = pron[pron["is_primary"] == "1"][["word__id", "yngping"]]
    id2text = dict(zip(word["id"], word["text"]))
    primary = primary.assign(text=primary["word__id"].map(id2text))
    primary = primary[primary["text"].notna() & primary["yngping"].notna()]
    primary = primary[primary["text"].str.len().between(1, 8)]
    primary["char_len"] = primary["text"].str.len()
    rng = random.Random(seed)
    sampled: list[dict] = []
    per_bucket = max(1, n // 4)
    for bucket in (1, 2, 3, 4):
        sub = primary[primary["char_len"] == bucket].reset_index(drop=True)
        if len(sub) == 0:
            continue
        take = min(per_bucket, len(sub))
        idx = rng.sample(range(len(sub)), take)
        for i in idx:
            row = sub.iloc[i]
            sampled.append({
                "text": row["text"],
                "gold_yngping": row["yngping"],
                "char_len": int(row["char_len"]),
            })
    rng.shuffle(sampled)
    return sampled[:n]


def make_request(custom_id: str, word: str, model: str, reasoning_effort: str | None, max_completion_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(word)},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort is not None and (model.startswith("gpt-5") or model.startswith("o")):
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


# --- subcommands -----------------------------------------------------------

def cmd_submit(args, client):
    eval_set = build_eval_set(args.resources_dir, args.n, args.seed)
    print(f"[batch] eval set: {len(eval_set)} words", file=sys.stderr)

    # Save the eval set so collect can re-join by custom_id.
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)
    with args.out_eval.open("w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"[batch] saved eval set to {args.out_eval}", file=sys.stderr)

    jsonl_path = args.out_eval.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(eval_set):
            req = make_request(
                f"g2p-{i:04d}", ex["text"], args.model,
                args.reasoning_effort, args.max_completion_tokens,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"[batch] wrote requests to {jsonl_path}  (reasoning={args.reasoning_effort}, max_completion_tokens={args.max_completion_tokens})", file=sys.stderr)

    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    print(f"[batch] uploaded file id={up.id}", file=sys.stderr)

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "fuzhoubench-g2p", "model": args.model, "n": str(args.n)},
    )
    print(f"[batch] created batch id={batch.id}  status={batch.status}", file=sys.stderr)
    print(batch.id)


def cmd_poll(args, client):
    b = client.batches.retrieve(args.batch_id)
    counts = b.request_counts
    print(json.dumps({
        "id": b.id,
        "status": b.status,
        "created_at": b.created_at,
        "in_progress_at": b.in_progress_at,
        "completed_at": b.completed_at,
        "request_counts": {"total": counts.total, "completed": counts.completed, "failed": counts.failed},
        "output_file_id": b.output_file_id,
        "error_file_id": b.error_file_id,
    }, indent=2))


def cmd_collect(args, client):
    b = client.batches.retrieve(args.batch_id)
    if b.status != "completed":
        print(f"[batch] not yet completed: status={b.status}", file=sys.stderr)
        sys.exit(2)
    print(f"[batch] downloading {b.output_file_id}", file=sys.stderr)
    out_text = client.files.content(b.output_file_id).text

    # Map custom_id -> response content
    cid_to_pred: dict[str, str] = {}
    for line in out_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj["custom_id"]
        try:
            content = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        cid_to_pred[cid] = content.strip()

    eval_set = json.loads(args.eval_set_file.read_text(encoding="utf-8"))
    results = []
    agg = {"n_syl_gold": 0, "n_syl_correct": 0, "n_tone_correct": 0, "n_word_exact": 0}
    for i, ex in enumerate(eval_set):
        pred = cid_to_pred.get(f"g2p-{i:04d}", "")
        sc = score_pair(pred, ex["gold_yngping"])
        word_exact = 1 if pred.strip() == ex["gold_yngping"].strip() else 0
        results.append({**ex, "pred_yngping": pred, **sc, "word_exact": word_exact})
        for k in ("n_syl_gold", "n_syl_correct", "n_tone_correct"):
            agg[k] += sc[k]
        agg["n_word_exact"] += word_exact

    by_len = {}
    for r in results:
        L = r["char_len"]
        by_len.setdefault(L, {"n": 0, "exact": 0, "syl_g": 0, "syl_c": 0})
        by_len[L]["n"] += 1
        by_len[L]["exact"] += r["word_exact"]
        by_len[L]["syl_g"] += r["n_syl_gold"]
        by_len[L]["syl_c"] += r["n_syl_correct"]
    by_len_out = {
        str(k): {"n": v["n"], "word_exact_acc": v["exact"] / max(1, v["n"]),
                 "syl_acc": v["syl_c"] / max(1, v["syl_g"])}
        for k, v in sorted(by_len.items())
    }
    summary = {
        "model": args.model,
        "n_words": len(results),
        "word_exact_accuracy": agg["n_word_exact"] / max(1, len(results)),
        "syllable_accuracy": agg["n_syl_correct"] / max(1, agg["n_syl_gold"]),
        "tone_accuracy": agg["n_tone_correct"] / max(1, agg["n_syl_gold"]),
        "by_char_len": by_len_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results, "batch_id": args.batch_id}, f, ensure_ascii=False, indent=2)
    print(f"[batch] wrote {args.out}", file=sys.stderr)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    len={kk}: {vv}")
        else:
            print(f"  {k}: {v}")


# --- main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key-file", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--model", required=True)
    p_sub.add_argument("--n", type=int, default=500)
    p_sub.add_argument("--seed", type=int, default=42)
    p_sub.add_argument("--resources-dir", type=Path, default=Path("foochow-server/resources"))
    p_sub.add_argument("--out-eval", type=Path, default=Path("papers/data/batch_eval_set.json"))
    p_sub.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="none",
                       help="Reasoning effort for GPT-5/o-series models. Default none (no reasoning).")
    p_sub.add_argument("--max-completion-tokens", type=int, default=128,
                       help="Cap on output tokens. Bump to 2048+ when using reasoning so the visible output "
                            "is not eaten by reasoning tokens.")

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("--batch-id", required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--batch-id", required=True)
    p_col.add_argument("--model", required=True)
    p_col.add_argument("--eval-set-file", type=Path, default=Path("papers/data/batch_eval_set.json"))
    p_col.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    from openai import OpenAI
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=api_key)

    if args.cmd == "submit":
        cmd_submit(args, client)
    elif args.cmd == "poll":
        cmd_poll(args, client)
    elif args.cmd == "collect":
        cmd_collect(args, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
