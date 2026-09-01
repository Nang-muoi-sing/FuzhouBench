"""
Sandhi prediction baseline via OpenAI Batch API.

Mirrors sandhi_baseline_remote.py (same eval set, same few-shot block,
same scoring) so OpenAI numbers are directly comparable to the remote
Qwen3 baselines. Subcommands: submit | poll | collect.

Run:
  python sandhi_baseline_openai_batch.py --api-key-file <path> \
    submit --model gpt-5.4-mini --n 500
  python sandhi_baseline_openai_batch.py --api-key-file <path> \
    poll --batch-id <id>
  python sandhi_baseline_openai_batch.py --api-key-file <path> \
    collect --batch-id <id> --model gpt-5.4-mini \
    --out papers/data/sandhi_gpt5.4-mini.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")


def parse_yngping(s: str) -> list[tuple[str, str]]:
    s = s.strip().lower().split("\n")[0]
    out = []
    for tok in s.split():
        m = YNGPING_TOKEN_RE.match(tok)
        if not m:
            continue
        out.append((m.group(1), m.group(2)))
    return out


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


def build_test_set(resources_dir: Path, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Same logic as sandhi_baseline_remote.build_test_set to keep results comparable."""
    feng = pd.read_csv(
        resources_dir / "FengResource.tsv", sep="\t", dtype=str,
        keep_default_na=False, na_values=[""], encoding="utf-8", engine="python",
    )
    df = feng[feng["literal_pron"].notna() & feng["sandhi_pron"].notna()].copy()
    df["lit_syl"] = df["literal_pron"].apply(lambda s: len(parse_yngping(s)))
    df["san_syl"] = df["sandhi_pron"].apply(lambda s: len(parse_yngping(s)))
    df = df[df["lit_syl"] == df["san_syl"]]
    df = df[df["lit_syl"].between(2, 5)].reset_index(drop=True)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    test_idx = sorted(idx[:n])
    train_idx = sorted(idx[n:])

    def row_to_dict(i: int) -> dict:
        r = df.iloc[i]
        return {
            "id": int(r["id"]),
            "literal_pron": r["literal_pron"],
            "sandhi_pron": r["sandhi_pron"],
            "syl_count": int(r["lit_syl"]),
        }

    return [row_to_dict(i) for i in test_idx], [row_to_dict(i) for i in train_idx]


SYSTEM_PROMPT = (
    "You are a Fuzhou Eastern Min phonology expert. Given a literal "
    "(citation-form) Yngping string for a multi-syllable word, output its "
    "surface (sandhi-form) Yngping string. Fuzhou tone sandhi mostly changes "
    "the tone digits of non-final syllables; segments stay the same. "
    "Tone digits come from {55, 53, 33, 21, 24, 242, 213, 5, 0}. "
    "Output ONLY the sandhi Yngping string, space-separated, no commentary."
)


def build_fewshot(train_pool: list[dict], k: int = 5, seed: int = 7) -> str:
    rng = random.Random(seed)
    candidates = [x for x in train_pool if x["literal_pron"] != x["sandhi_pron"] and 2 <= x["syl_count"] <= 3]
    rng.shuffle(candidates)
    picks = candidates[:k]
    return "\n".join(f"Literal: {x['literal_pron']}\nSandhi: {x['sandhi_pron']}" for x in picks)


def index_by_context_pattern(train_pool: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for ex in train_pool:
        lit = parse_yngping(ex["literal_pron"])
        if len(lit) < 2:
            continue
        for i in range(len(lit) - 1):
            key = (lit[i][1], lit[i + 1][1])
            out.setdefault(key, []).append(ex)
    return out


def first_non_final_pattern(literal: str) -> tuple[str, str] | None:
    syls = parse_yngping(literal)
    if len(syls) < 2:
        return None
    return (syls[0][1], syls[1][1])


def build_pattern_matched_fewshot(
    literal: str,
    pattern_index: dict[tuple[str, str], list[dict]],
    train_pool: list[dict],
    k: int = 5,
    seed: int = 13,
) -> str:
    pattern = first_non_final_pattern(literal)
    rng = random.Random(seed ^ hash(literal))
    if pattern is None:
        cands = list(train_pool); rng.shuffle(cands); picks = cands[:k]
    else:
        same = list(pattern_index.get(pattern, [])); rng.shuffle(same); picks = same[:k]
        if len(picks) < k:
            backup = [x for x in train_pool if x not in picks]
            rng.shuffle(backup); picks = picks + backup[:(k - len(picks))]
    return "\n".join(f"Literal: {x['literal_pron']}\nSandhi: {x['sandhi_pron']}" for x in picks)


def build_user_prompt(literal: str, fewshot: str) -> str:
    return f"Examples:\n{fewshot}\n\nLiteral: {literal}\nSandhi:"


def make_request(custom_id: str, literal: str, fewshot: str, model: str,
                 reasoning_effort: str | None, max_completion_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(literal, fewshot)},
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


def cmd_submit(args, client):
    test_set, train_pool = build_test_set(args.resources_dir, args.n, args.seed)
    print(f"[batch] test={len(test_set)}  train_pool={len(train_pool)}", file=sys.stderr)

    if args.fewshot_mode == "random":
        shared = build_fewshot(train_pool)
        print(f"[batch] shared random few-shot:\n{shared}\n", file=sys.stderr)
        fewshots = [shared] * len(test_set)
    else:
        idx = index_by_context_pattern(train_pool)
        print(f"[batch] indexed {len(idx)} non-final (curr,next) patterns", file=sys.stderr)
        fewshots = [
            build_pattern_matched_fewshot(ex["literal_pron"], idx, train_pool)
            for ex in test_set
        ]
        for j in range(min(2, len(test_set))):
            print(f"[batch] per-item fewshot for test[{j}] literal={test_set[j]['literal_pron']!r}\n{fewshots[j]}\n", file=sys.stderr)

    args.out_eval.parent.mkdir(parents=True, exist_ok=True)
    with args.out_eval.open("w", encoding="utf-8") as f:
        json.dump({"test_set": test_set, "fewshots": fewshots, "fewshot_mode": args.fewshot_mode}, f, ensure_ascii=False, indent=2)

    jsonl_path = args.out_eval.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(test_set):
            req = make_request(
                f"sandhi-{i:04d}", ex["literal_pron"], fewshots[i], args.model,
                args.reasoning_effort, args.max_completion_tokens,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"[batch] wrote requests to {jsonl_path}  "
          f"(reasoning={args.reasoning_effort}, max_completion_tokens={args.max_completion_tokens})",
          file=sys.stderr)

    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    print(f"[batch] uploaded file id={up.id}", file=sys.stderr)
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "fuzhoubench-sandhi", "model": args.model, "n": str(args.n)},
    )
    print(f"[batch] created batch id={batch.id}  status={batch.status}", file=sys.stderr)
    print(batch.id)


def cmd_poll(args, client):
    b = client.batches.retrieve(args.batch_id)
    counts = b.request_counts
    print(json.dumps({
        "id": b.id,
        "status": b.status,
        "request_counts": {"total": counts.total, "completed": counts.completed, "failed": counts.failed},
        "output_file_id": b.output_file_id,
        "error_file_id": b.error_file_id,
    }, indent=2))


def cmd_collect(args, client):
    b = client.batches.retrieve(args.batch_id)
    if b.status != "completed":
        print(f"[batch] not yet completed: status={b.status}", file=sys.stderr)
        sys.exit(2)
    out_text = client.files.content(b.output_file_id).text
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

    stash = json.loads(args.eval_set_file.read_text(encoding="utf-8"))
    test_set = stash["test_set"]

    results = []
    agg = {"n_syl_gold": 0, "n_syl_correct": 0, "n_tone_correct": 0, "n_word_exact": 0}
    for i, ex in enumerate(test_set):
        pred = cid_to_pred.get(f"sandhi-{i:04d}", "")
        sc = score_pair(pred, ex["sandhi_pron"])
        word_exact = 1 if pred.strip() == ex["sandhi_pron"].strip() else 0
        results.append({**ex, "pred_sandhi": pred, **sc, "word_exact": word_exact})
        for k in ("n_syl_gold", "n_syl_correct", "n_tone_correct"):
            agg[k] += sc[k]
        agg["n_word_exact"] += word_exact

    by_syl = {}
    for r in results:
        s = r["syl_count"]
        d = by_syl.setdefault(s, {"n": 0, "exact": 0, "syl_g": 0, "syl_c": 0})
        d["n"] += 1
        d["exact"] += r["word_exact"]
        d["syl_g"] += r["n_syl_gold"]
        d["syl_c"] += r["n_syl_correct"]
    by_syl_out = {
        str(k): {"n": v["n"], "word_exact_acc": v["exact"] / max(1, v["n"]),
                 "syl_acc": v["syl_c"] / max(1, v["syl_g"])}
        for k, v in sorted(by_syl.items())
    }
    summary = {
        "label": args.model,
        "n_words": len(results),
        "word_exact_accuracy": agg["n_word_exact"] / max(1, len(results)),
        "syllable_accuracy": agg["n_syl_correct"] / max(1, agg["n_syl_gold"]),
        "tone_accuracy": agg["n_tone_correct"] / max(1, agg["n_syl_gold"]),
        "by_syl_count": by_syl_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results, "batch_id": args.batch_id,
                   "fewshot_mode": stash.get("fewshot_mode", "random")},
                  f, ensure_ascii=False, indent=2)
    print(f"[batch] wrote {args.out}", file=sys.stderr)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--model", required=True)
    p_sub.add_argument("--n", type=int, default=500)
    p_sub.add_argument("--seed", type=int, default=42)
    p_sub.add_argument("--resources-dir", type=Path, default=Path("foochow-server/resources"))
    p_sub.add_argument("--out-eval", type=Path, default=Path("papers/data/sandhi_batch_eval_set.json"))
    p_sub.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="none")
    p_sub.add_argument("--max-completion-tokens", type=int, default=128)
    p_sub.add_argument("--fewshot-mode", choices=["random", "pattern"], default="random")

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("--batch-id", required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("--batch-id", required=True)
    p_col.add_argument("--model", required=True)
    p_col.add_argument("--eval-set-file", type=Path, default=Path("papers/data/sandhi_batch_eval_set.json"))
    p_col.add_argument("--out", type=Path, required=True)

    args = p.parse_args()
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
    raise SystemExit(main())
