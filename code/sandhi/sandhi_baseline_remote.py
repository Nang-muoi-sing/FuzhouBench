"""
Sandhi-prediction baseline runner for FuzhouBench, runs end-to-end on the
remote H100. Mirrors g2p_baseline_remote.py: builds a stratified test
sample, prompts a local Qwen model, scores syllable / tone / word accuracy.

Task: given a literal (citation-form) Yngping string, predict the surface
(sandhi-form) Yngping.

Data source: FengResource.tsv columns `literal_pron`, `sandhi_pron`.

Also supports two non-LLM baselines via --baseline:
  identity   : predict sandhi = literal verbatim (sanity floor).
  mode-rule  : apply the per-tone mode rule observed in FengResource (the
               most frequent target tone for each literal tone in non-final
               position); final syllable's tone is kept unchanged.

Run on remote:
  <PROJECT_ROOT>/venv/bin/python \
    <PROJECT_ROOT>/scripts/sandhi_baseline_remote.py \
    --baseline identity --resources-dir <PROJECT_ROOT>/resources \
    --out <PROJECT_ROOT>/papers-data/sandhi_identity.json --n 500

  ... --baseline llm --model-path <hf-path> --model-label Qwen3-8B ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
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


def join_yngping(syls: list[tuple[str, str]]) -> str:
    return " ".join(f"{seg}{tone}" for seg, tone in syls)


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


# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

def build_test_set(resources_dir: Path, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (test_set, train_pool) where train_pool feeds the mode-rule
    learner and the LLM few-shot prompt."""
    feng = pd.read_csv(
        resources_dir / "FengResource.tsv", sep="\t", dtype=str,
        keep_default_na=False, na_values=[""], encoding="utf-8", engine="python",
    )
    df = feng[feng["literal_pron"].notna() & feng["sandhi_pron"].notna()].copy()
    df["lit_syl"] = df["literal_pron"].apply(lambda s: len(parse_yngping(s)))
    df["san_syl"] = df["sandhi_pron"].apply(lambda s: len(parse_yngping(s)))
    df = df[df["lit_syl"] == df["san_syl"]]
    # Restrict to multi-syllable so sandhi is meaningful.
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

    test_set = [row_to_dict(i) for i in test_idx]
    train_pool = [row_to_dict(i) for i in train_idx]
    return test_set, train_pool


# ---------------------------------------------------------------------------
# Mode-rule predictor
# ---------------------------------------------------------------------------

def learn_mode_rule(train_pool: list[dict]) -> dict[str, str]:
    """Build a literal_tone -> majority sandhi_tone map from the non-final
    syllables of the train pool."""
    counts: dict[str, Counter] = {}
    for ex in train_pool:
        lit = parse_yngping(ex["literal_pron"])
        san = parse_yngping(ex["sandhi_pron"])
        if len(lit) != len(san) or len(lit) < 2:
            continue
        for i in range(len(lit) - 1):  # non-final
            lt = lit[i][1]
            st = san[i][1]
            counts.setdefault(lt, Counter())[st] += 1
    rule_map = {lt: c.most_common(1)[0][0] for lt, c in counts.items()}
    return rule_map


def learn_context_rule(train_pool: list[dict]) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Build a (literal_t_curr, literal_t_next) -> majority sandhi_t_curr map.
    Also returns a single-tone fallback (the mode-rule) for unseen contexts."""
    ctx_counts: dict[tuple[str, str], Counter] = {}
    for ex in train_pool:
        lit = parse_yngping(ex["literal_pron"])
        san = parse_yngping(ex["sandhi_pron"])
        if len(lit) != len(san) or len(lit) < 2:
            continue
        for i in range(len(lit) - 1):
            key = (lit[i][1], lit[i + 1][1])
            ctx_counts.setdefault(key, Counter())[san[i][1]] += 1
    ctx_rule = {k: c.most_common(1)[0][0] for k, c in ctx_counts.items()}
    fallback = learn_mode_rule(train_pool)
    return ctx_rule, fallback


def apply_rule(literal: str, rule_map: dict[str, str]) -> str:
    syls = parse_yngping(literal)
    if not syls:
        return literal
    out = []
    for i, (seg, tone) in enumerate(syls):
        if i < len(syls) - 1:
            new_tone = rule_map.get(tone, tone)
            out.append((seg, new_tone))
        else:
            out.append((seg, tone))
    return join_yngping(out)


def apply_context_rule(literal: str,
                       ctx_rule: dict[tuple[str, str], str],
                       fallback: dict[str, str]) -> str:
    syls = parse_yngping(literal)
    if not syls:
        return literal
    out = []
    for i, (seg, tone) in enumerate(syls):
        if i < len(syls) - 1:
            next_tone = syls[i + 1][1]
            new_tone = ctx_rule.get((tone, next_tone), fallback.get(tone, tone))
            out.append((seg, new_tone))
        else:
            out.append((seg, tone))
    return join_yngping(out)


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Fuzhou Eastern Min phonology expert. Given a literal "
    "(citation-form) Yngping string for a multi-syllable word, output its "
    "surface (sandhi-form) Yngping string. Fuzhou tone sandhi mostly changes "
    "the tone digits of non-final syllables; segments stay the same. "
    "Tone digits come from {55, 53, 33, 21, 24, 242, 213, 5, 0}. "
    "Output ONLY the sandhi Yngping string, space-separated, no commentary."
)


def build_fewshot(train_pool: list[dict], k: int = 5, seed: int = 7) -> str:
    """Random few-shot block: K examples drawn from the train pool."""
    rng = random.Random(seed)
    candidates = [x for x in train_pool if x["literal_pron"] != x["sandhi_pron"] and 2 <= x["syl_count"] <= 3]
    rng.shuffle(candidates)
    picks = candidates[:k]
    return "\n".join(f"Literal: {x['literal_pron']}\nSandhi: {x['sandhi_pron']}" for x in picks)


def index_by_context_pattern(train_pool: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Index train examples by every non-final (curr, next) tone pair they contain."""
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
    """Build a per-item few-shot block whose K examples all share the test item's
    first non-final (curr, next) tone pattern. If fewer than K such examples exist,
    pad with random train examples (clearly marked at the end)."""
    pattern = first_non_final_pattern(literal)
    rng = random.Random(seed ^ hash(literal))
    if pattern is None:
        candidates = list(train_pool)
        rng.shuffle(candidates)
        picks = candidates[:k]
    else:
        same = list(pattern_index.get(pattern, []))
        rng.shuffle(same)
        picks = same[:k]
        if len(picks) < k:
            backup = [x for x in train_pool if x not in picks]
            rng.shuffle(backup)
            picks = picks + backup[: (k - len(picks))]
    return "\n".join(f"Literal: {x['literal_pron']}\nSandhi: {x['sandhi_pron']}" for x in picks)


def build_user_prompt(literal: str, fewshot: str) -> str:
    return f"Examples:\n{fewshot}\n\nLiteral: {literal}\nSandhi:"


def run_llm_inference(
    model_path: str,
    eval_set: list[dict],
    fewshot_per_item: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    """fewshot_per_item[i] is the few-shot block to use for eval_set[i].
    For random-mode few-shot, pass [shared_block] * len(eval_set)."""
    assert len(fewshot_per_item) == len(eval_set)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[sandhi] loading tokenizer + model from {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    template_kwargs = {}
    try:
        _ = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        template_kwargs["enable_thinking"] = False
    except TypeError:
        pass

    results: list[dict] = []
    t0 = time.time()
    for i in range(0, len(eval_set), batch_size):
        batch = eval_set[i : i + batch_size]
        batch_fewshot = fewshot_per_item[i : i + batch_size]
        prompts = []
        for ex, fs in zip(batch, batch_fewshot):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ex["literal_pron"], fs)},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
            )
            prompts.append(text)
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for ex, fs, pred in zip(batch, batch_fewshot, decoded):
            pred_clean = pred.strip()
            sc = score_pair(pred_clean, ex["sandhi_pron"])
            results.append({
                **ex, "pred_sandhi": pred_clean, **sc,
                "word_exact": 1 if pred_clean == ex["sandhi_pron"].strip() else 0,
                "fewshot": fs,
            })
        elapsed = time.time() - t0
        done = i + len(batch)
        print(f"[sandhi] {done}/{len(eval_set)} ({done/max(elapsed,1e-6):.1f} qps, {elapsed:.0f}s)", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[dict], label: str) -> dict:
    n = len(results)
    n_syl_gold = sum(r["n_syl_gold"] for r in results)
    n_syl_correct = sum(r["n_syl_correct"] for r in results)
    n_tone_correct = sum(r["n_tone_correct"] for r in results)
    n_word_exact = sum(r["word_exact"] for r in results)
    by_syl = {}
    for r in results:
        s = r["syl_count"]
        d = by_syl.setdefault(s, {"n": 0, "exact": 0, "syl_g": 0, "syl_c": 0})
        d["n"] += 1
        d["exact"] += r["word_exact"]
        d["syl_g"] += r["n_syl_gold"]
        d["syl_c"] += r["n_syl_correct"]
    by_syl_out = {
        str(k): {"n": v["n"],
                 "word_exact_acc": v["exact"] / max(1, v["n"]),
                 "syl_acc": v["syl_c"] / max(1, v["syl_g"])}
        for k, v in sorted(by_syl.items())
    }
    return {
        "label": label,
        "n_words": n,
        "word_exact_accuracy": n_word_exact / max(1, n),
        "syllable_accuracy": n_syl_correct / max(1, n_syl_gold),
        "tone_accuracy": n_tone_correct / max(1, n_syl_gold),
        "by_syl_count": by_syl_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", choices=["identity", "mode-rule", "context-rule", "llm"], required=True)
    p.add_argument("--model-path", default=None, help="Required for --baseline llm.")
    p.add_argument("--model-label", default=None, help="Display label for the model.")
    p.add_argument("--resources-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--fewshot-mode", choices=["random", "pattern"], default="random",
                   help="random: shared 5-shot block. pattern: per-item 5-shot, all examples share the test item's first non-final (curr,next) tone pattern.")
    args = p.parse_args()

    test_set, train_pool = build_test_set(args.resources_dir, args.n, args.seed)
    print(f"[sandhi] test={len(test_set)}  train_pool={len(train_pool)}", file=sys.stderr)

    if args.baseline == "identity":
        results = []
        for ex in test_set:
            pred = ex["literal_pron"]
            sc = score_pair(pred, ex["sandhi_pron"])
            results.append({**ex, "pred_sandhi": pred, **sc,
                            "word_exact": 1 if pred.strip() == ex["sandhi_pron"].strip() else 0})
        summary = summarize(results, "identity")

    elif args.baseline == "mode-rule":
        rule_map = learn_mode_rule(train_pool)
        print(f"[sandhi] learned rule map: {rule_map}", file=sys.stderr)
        results = []
        for ex in test_set:
            pred = apply_rule(ex["literal_pron"], rule_map)
            sc = score_pair(pred, ex["sandhi_pron"])
            results.append({**ex, "pred_sandhi": pred, **sc,
                            "word_exact": 1 if pred.strip() == ex["sandhi_pron"].strip() else 0})
        summary = summarize(results, "mode-rule")
        summary["rule_map"] = rule_map

    elif args.baseline == "context-rule":
        ctx_rule, fallback = learn_context_rule(train_pool)
        print(f"[sandhi] context rule entries: {len(ctx_rule)}", file=sys.stderr)
        print(f"[sandhi] fallback (1-tone mode): {fallback}", file=sys.stderr)
        results = []
        for ex in test_set:
            pred = apply_context_rule(ex["literal_pron"], ctx_rule, fallback)
            sc = score_pair(pred, ex["sandhi_pron"])
            results.append({**ex, "pred_sandhi": pred, **sc,
                            "word_exact": 1 if pred.strip() == ex["sandhi_pron"].strip() else 0})
        summary = summarize(results, "context-rule")
        # Serialize the context rule with tuple keys flattened to strings.
        summary["ctx_rule"] = {f"{a},{b}": v for (a, b), v in ctx_rule.items()}
        summary["fallback_rule"] = fallback

    elif args.baseline == "llm":
        if not args.model_path or not args.model_label:
            p.error("--model-path and --model-label are required for --baseline llm")
        if args.fewshot_mode == "random":
            shared = build_fewshot(train_pool)
            print(f"[sandhi] shared random few-shot block:\n{shared}\n", file=sys.stderr)
            fewshot_per_item = [shared] * len(test_set)
        elif args.fewshot_mode == "pattern":
            idx = index_by_context_pattern(train_pool)
            print(f"[sandhi] indexed {len(idx)} non-final (curr,next) patterns in train pool", file=sys.stderr)
            fewshot_per_item = [
                build_pattern_matched_fewshot(ex["literal_pron"], idx, train_pool)
                for ex in test_set
            ]
            # Log the first three examples for reproducibility.
            for j in range(min(3, len(test_set))):
                print(f"[sandhi] per-item fewshot for test[{j}] literal={test_set[j]['literal_pron']!r}\n{fewshot_per_item[j]}\n", file=sys.stderr)
        else:
            p.error(f"unknown fewshot_mode={args.fewshot_mode}")
        results = run_llm_inference(args.model_path, test_set, fewshot_per_item,
                                    args.batch_size, args.max_new_tokens)
        summary = summarize(results, args.model_label)
        summary["fewshot_mode"] = args.fewshot_mode

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[sandhi] wrote {args.out}", file=sys.stderr)
    print("\nSUMMARY")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
