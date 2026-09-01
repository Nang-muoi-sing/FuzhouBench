"""
G2P baseline runner for FuzhouBench, designed to run on the remote H100 server.

Self-contained: reads TSVs, loads a local Qwen model via transformers,
runs batched inference, scores syllable / tone / word accuracy, writes JSON.

Run on the remote (after `source venv/bin/activate`):
  python g2p_baseline_remote.py \
    --model-path <HF_CACHE>/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --model-label Qwen3-1.7B \
    --resources-dir <PROJECT_ROOT>/foochow-server/resources \
    --out <OUT_DIR>/g2p_qwen3_1.7b.json \
    --n 500 --batch-size 16

Set CUDA_VISIBLE_DEVICES upstream to pick the GPU.
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FEWSHOT_EXAMPLES = [
    ("牛乳",   "ngu33 neing53"),      # 'cow-milk' = milk; primary reading (sandhi)
    ("講",     "goung33"),             # 'to speak'; single-syl primary
    ("白菜",   "bah21 jai213"),        # 'Chinese cabbage'; primary reading (sandhi, c->j)
    ("做什乇", "zo21 sie53 nooh24"), # 'do-what-thing' = to do what; primary (native speaker)
    ("我",     "nguai33"),             # 'I / me'; single-syl primary
]

SYSTEM_PROMPT = (
    "You convert Fuzhou Eastern Min words written in Chinese characters "
    "to their Yngping romanization. Each syllable in Yngping is letters "
    "followed by tone digits (one of: 55, 53, 33, 21, 24, 242, 213, 5, 0). "
    "Multi-syllable outputs are space-separated. "
    "Output ONLY the Yngping string, no commentary, no extra words."
)


def build_user_prompt(word: str) -> str:
    ex = "\n".join(f"Word: {w}\nYngping: {y}" for w, y in FEWSHOT_EXAMPLES)
    return f"Examples:\n{ex}\n\nWord: {word}\nYngping:"


YNGPING_TOKEN_RE = re.compile(r"([a-z]+)(\d+)")


def parse_yngping(s: str) -> list[tuple[str, str]]:
    s = s.strip().lower()
    # Stop at first newline or non-Yngping junk.
    s = s.split("\n")[0]
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


def run_inference(
    model_path: str,
    eval_set: list[dict],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    print(f"[g2p] loading tokenizer from {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[g2p] loading model {model_path}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Qwen3 chat template: disable thinking via the chat-template flag.
    template_kwargs = {}
    try:
        # Probe whether the template supports enable_thinking
        _ = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        template_kwargs["enable_thinking"] = False
    except TypeError:
        pass

    results: list[dict] = []
    t0 = time.time()
    for i in range(0, len(eval_set), batch_size):
        batch = eval_set[i : i + batch_size]
        prompts = []
        for ex in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ex["text"])},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
            )
            prompts.append(text)

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Slice off the prompt portion.
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for ex, pred in zip(batch, decoded):
            pred_clean = pred.strip()
            sc = score_pair(pred_clean, ex["gold_yngping"])
            results.append({
                **ex,
                "pred_yngping": pred_clean,
                **sc,
                "word_exact": 1 if pred_clean == ex["gold_yngping"].strip() else 0,
            })

        elapsed = time.time() - t0
        done = i + len(batch)
        rate = done / max(elapsed, 1e-6)
        print(f"[g2p] {done}/{len(eval_set)} done ({rate:.1f} qps, {elapsed:.0f}s)", file=sys.stderr)

    return results


def summarize(results: list[dict], model_label: str) -> dict:
    n = len(results)
    n_syl_gold = sum(r["n_syl_gold"] for r in results)
    n_syl_correct = sum(r["n_syl_correct"] for r in results)
    n_tone_correct = sum(r["n_tone_correct"] for r in results)
    n_word_exact = sum(r["word_exact"] for r in results)

    by_len = {}
    for r in results:
        L = r["char_len"]
        by_len.setdefault(L, {"n": 0, "exact": 0, "syl_g": 0, "syl_c": 0})
        by_len[L]["n"] += 1
        by_len[L]["exact"] += r["word_exact"]
        by_len[L]["syl_g"] += r["n_syl_gold"]
        by_len[L]["syl_c"] += r["n_syl_correct"]
    by_len_out = {
        str(k): {
            "n": v["n"],
            "word_exact_acc": v["exact"] / max(1, v["n"]),
            "syl_acc": v["syl_c"] / max(1, v["syl_g"]),
        }
        for k, v in sorted(by_len.items())
    }
    return {
        "model": model_label,
        "n_words": n,
        "word_exact_accuracy": n_word_exact / max(1, n),
        "syllable_accuracy": n_syl_correct / max(1, n_syl_gold),
        "tone_accuracy": n_tone_correct / max(1, n_syl_gold),
        "by_char_len": by_len_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--resources-dir", type=Path,
                        default=Path("<PROJECT_ROOT>/foochow-server/resources"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    eval_set = build_eval_set(args.resources_dir, args.n, args.seed)
    print(f"[g2p] eval set: {len(eval_set)} words", file=sys.stderr)

    results = run_inference(args.model_path, eval_set, args.batch_size, args.max_new_tokens)
    summary = summarize(results, args.model_label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[g2p] wrote {args.out}", file=sys.stderr)
    print("\nSUMMARY")
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
