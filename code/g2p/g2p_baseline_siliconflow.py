"""G2P baseline via SiliconFlow (OpenAI-compatible API).

Runs a Qwen3-* (or any SiliconFlow-hosted) model on the same 500-word G2P
test sample as the OpenAI batch script. Uses concurrent synchronous
requests (no batch API on SiliconFlow), reusing the eval-set builder,
prompt, and scoring from `g2p_baseline_openai_batch.py`.

Usage:
  python g2p_baseline_siliconflow.py \
      --api-key-file .siliconflow_key \
      --model Qwen/Qwen3-8B --model-label Qwen3-8B \
      --eval-set-file papers/data/batch_eval_set_v2.json \
      --out papers/data/g2p_qwen3_8b_siliconflow.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

# Reuse eval-set + scoring + prompt from the OpenAI batch script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2p_baseline_openai_batch import (  # type: ignore
    SYSTEM_PROMPT, build_user_prompt, build_eval_set, score_pair,
)

BASE_URL = "https://api.siliconflow.cn/v1"


def call_once(client, model: str, word: str, max_tokens: int = 40) -> str:
    prompt = build_user_prompt(word)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    if "qwen3" in model.lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def run_one(client, model: str, i: int, ex: dict, max_tokens: int, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            pred = call_once(client, model, ex["text"], max_tokens)
            return (i, ex, pred, None)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return (i, ex, "", str(last_err))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--model-label", default="Qwen3-8B")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--resources-dir", type=Path, default=Path("foochow-server/resources"))
    p.add_argument("--eval-set-file", type=Path, default=None,
                   help="Pre-built eval set JSON (skip rebuild if provided).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openai import OpenAI
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    if args.eval_set_file and args.eval_set_file.is_file():
        eval_set = json.loads(args.eval_set_file.read_text(encoding="utf-8"))
        print(f"[sf] loaded eval set from {args.eval_set_file} ({len(eval_set)} words)", file=sys.stderr)
    else:
        eval_set = build_eval_set(args.resources_dir, args.n, args.seed)
        print(f"[sf] built fresh eval set ({len(eval_set)} words, seed={args.seed})", file=sys.stderr)

    print(f"[sf] model={args.model}  concurrency={args.concurrency}", file=sys.stderr)
    t0 = time.time()
    results: list = [None] * len(eval_set)
    n_done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run_one, client, args.model, i, ex, args.max_tokens)
                for i, ex in enumerate(eval_set)]
        for fut in cf.as_completed(futs):
            i, ex, pred, err = fut.result()
            sc = score_pair(pred, ex["gold_yngping"])
            word_exact = 1 if pred.strip() == ex["gold_yngping"].strip() else 0
            results[i] = {**ex, "pred_yngping": pred, **sc, "word_exact": word_exact, "error": err}
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                print(f"[sf] {n_done}/{len(eval_set)} done  ({elapsed:.0f}s elapsed)", file=sys.stderr)

    n_err = sum(1 for r in results if r["error"])
    total_syl_gold = sum(r["n_syl_gold"] for r in results)
    total_syl_correct = sum(r["n_syl_correct"] for r in results)
    total_tone_correct = sum(r["n_tone_correct"] for r in results)
    word_exact_acc = sum(r["word_exact"] for r in results) / len(results)
    syl_acc = total_syl_correct / total_syl_gold if total_syl_gold else 0
    tone_acc = total_tone_correct / total_syl_gold if total_syl_gold else 0

    by_len: dict[int, dict] = {}
    for r in results:
        L = r["char_len"]
        d = by_len.setdefault(L, {"n": 0, "exact": 0, "syl_g": 0, "syl_c": 0})
        d["n"] += 1
        d["exact"] += r["word_exact"]
        d["syl_g"] += r["n_syl_gold"]
        d["syl_c"] += r["n_syl_correct"]
    by_char_len = {
        L: {"n": v["n"],
            "word_exact_acc": v["exact"] / v["n"],
            "syl_acc": v["syl_c"] / v["syl_g"] if v["syl_g"] else 0}
        for L, v in sorted(by_len.items())
    }

    summary = {
        "model_label": args.model_label,
        "model_id": args.model,
        "backend": "siliconflow",
        "n_words": len(results),
        "n_errors": n_err,
        "elapsed_sec": round(time.time() - t0, 1),
        "word_exact_accuracy": word_exact_acc,
        "syllable_accuracy": syl_acc,
        "tone_accuracy": tone_acc,
        "by_char_len": by_char_len,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[sf] wrote {args.out}", file=sys.stderr)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    len={kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
