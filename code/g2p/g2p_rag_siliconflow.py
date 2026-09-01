"""Retrieval-augmented G2P baseline (reviewer-suggested upper bound).

For each test word, retrieve up to K training words that share characters
with it, and use their (characters, Yngping) pairs as dynamic few-shot
demonstrations in place of the 5 fixed ones. Tests whether poor few-shot
G2P reflects "untrained on Yngping" rather than "unlearnable": if
in-domain demonstrations lift accuracy substantially, the orthography is
learnable from data the release already contains.

Retrieval is deliberately simple (character overlap, no embeddings):
  score(train_word) = |chars(test) ∩ chars(train)| / |chars(test)|
  tie-break: prefer words of similar length, then more frequent chars.
Training pool = all word--primary-reading pairs EXCLUDING the 500 test
words (same exclusion seed as the eval set).

Usage:
  python papers/scripts/g2p_rag_siliconflow.py \
      --api-key-file .siliconflow_key \
      --model Qwen/Qwen3-8B --model-label Qwen3-8B-RAG \
      --eval-set-file papers/data/batch_eval_set_v2.json \
      --out papers/data/g2p_rag_qwen3_8b_sf.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2p_baseline_openai_batch import SYSTEM_PROMPT, score_pair  # type: ignore

BASE_URL = "https://api.siliconflow.cn/v1"


def load_train_pool(resources_dir: Path, test_words: set[str]) -> list[tuple[str, str]]:
    pron = pd.read_csv(resources_dir / "PronunciationResource.tsv", sep="\t", dtype=str,
                       keep_default_na=False, na_values=[""], encoding="utf-8", engine="python")
    word = pd.read_csv(resources_dir / "WordResource.tsv", sep="\t", dtype=str,
                       keep_default_na=False, na_values=[""], encoding="utf-8", engine="python")
    primary = pron[pron["is_primary"] == "1"][["word__id", "yngping"]]
    id2text = dict(zip(word["id"], word["text"]))
    primary = primary.assign(text=primary["word__id"].map(id2text))
    primary = primary[primary["text"].notna() & primary["yngping"].notna()]
    primary = primary[primary["text"].str.len().between(1, 8)]
    pool = [(r["text"], r["yngping"]) for _, r in primary.iterrows()
            if r["text"] not in test_words]
    return pool


def build_char_index(pool: list[tuple[str, str]]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = defaultdict(list)
    for i, (text, _) in enumerate(pool):
        for ch in set(text):
            idx[ch].append(i)
    return idx


def retrieve(test_word: str, pool: list[tuple[str, str]],
             char_idx: dict[str, list[int]], k: int) -> list[tuple[str, str]]:
    cand_scores: dict[int, float] = defaultdict(float)
    tchars = set(test_word)
    for ch in tchars:
        for i in char_idx.get(ch, []):
            cand_scores[i] += 1.0
    if not cand_scores:
        return []
    ranked = sorted(
        cand_scores.items(),
        key=lambda kv: (-kv[1] / len(tchars),
                        abs(len(pool[kv[0]][0]) - len(test_word)),
                        len(pool[kv[0]][0])))
    return [pool[i] for i, _ in ranked[:k]]


def build_prompt(word: str, demos: list[tuple[str, str]]) -> str:
    ex = "\n".join(f"Word: {w}\nYngping: {y}" for w, y in demos)
    return f"Examples:\n{ex}\n\nWord: {word}\nYngping:"


def call_once(client, model: str, prompt: str, max_tokens: int = 40) -> str:
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0,
    )
    if "qwen3" in model.lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def run_one(client, model, i, ex, demos, max_tokens, retries=3):
    prompt = build_prompt(ex["text"], demos)
    last = None
    for a in range(retries):
        try:
            return i, call_once(client, model, prompt, max_tokens), None
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    return i, "", str(last)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--model-label", default="Qwen3-8B-RAG")
    p.add_argument("--k", type=int, default=8, help="retrieved demonstrations per test word")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--resources-dir", type=Path, default=Path("foochow-server/resources"))
    p.add_argument("--eval-set-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key_file.read_text(encoding="utf-8").strip(),
                    base_url=BASE_URL)

    eval_set = json.loads(args.eval_set_file.read_text(encoding="utf-8"))
    test_words = {ex["text"] for ex in eval_set}
    pool = load_train_pool(args.resources_dir, test_words)
    char_idx = build_char_index(pool)
    print(f"[rag] eval={len(eval_set)}  train pool={len(pool)}", file=sys.stderr)

    demos_per_item = [retrieve(ex["text"], pool, char_idx, args.k) for ex in eval_set]
    n_zero = sum(1 for d in demos_per_item if not d)
    cov = sum(len(set(ex['text']) & set(''.join(w for w, _ in d))) / len(set(ex['text']))
              for ex, d in zip(eval_set, demos_per_item)) / len(eval_set)
    print(f"[rag] items with zero retrieved demos: {n_zero}; "
          f"mean char coverage by demos: {cov:.2%}", file=sys.stderr)

    t0 = time.time()
    results: list = [None] * len(eval_set)
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex_pool:
        futs = [ex_pool.submit(run_one, client, args.model, i, ex, demos_per_item[i], args.max_tokens)
                for i, ex in enumerate(eval_set)]
        done = 0
        for fut in cf.as_completed(futs):
            i, pred, err = fut.result()
            ex = eval_set[i]
            sc = score_pair(pred, ex["gold_yngping"])
            we = 1 if pred.strip() == ex["gold_yngping"].strip() else 0
            results[i] = {**ex, "pred_yngping": pred, **sc, "word_exact": we,
                          "n_demos": len(demos_per_item[i]), "error": err}
            done += 1
            if done % 50 == 0:
                print(f"[rag] {done}/{len(eval_set)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    n_err = sum(1 for r in results if r["error"])
    sg = sum(r["n_syl_gold"] for r in results)
    summary = {
        "model_label": args.model_label, "model_id": args.model,
        "backend": "siliconflow", "k_retrieved": args.k,
        "n_words": len(results), "n_errors": n_err,
        "mean_char_coverage": cov,
        "elapsed_sec": round(time.time() - t0, 1),
        "word_exact_accuracy": sum(r["word_exact"] for r in results) / len(results),
        "syllable_accuracy": sum(r["n_syl_correct"] for r in results) / sg,
        "tone_accuracy": sum(r["n_tone_correct"] for r in results) / sg,
    }
    by_len: dict[int, dict] = {}
    for r in results:
        d = by_len.setdefault(r["char_len"], {"n": 0, "exact": 0})
        d["n"] += 1
        d["exact"] += r["word_exact"]
    summary["by_char_len"] = {k: {"n": v["n"], "word_exact_acc": v["exact"] / v["n"]}
                              for k, v in sorted(by_len.items())}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rag] wrote {args.out}", file=sys.stderr)
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
